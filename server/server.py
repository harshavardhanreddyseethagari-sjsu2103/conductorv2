# server/server.py — Conductor v2
#
# Key changes from v1:
#   1. Resource-aware claiming: workers declare GPU/mem capacity;
#      claim query only matches tasks whose requirements fit.
#   2. Push instead of poll: workers hold a persistent WebSocket
#      connection; server pushes "task_available" when a task enters
#      the queue, so workers act immediately instead of waiting up to
#      POLL_INTERVAL seconds.
#
# Run:
#   uvicorn server.server:app --reload --port 8001

import os
import json
import asyncio
from contextlib import asynccontextmanager
from datetime import datetime, timezone, timedelta

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from pydantic import BaseModel
import psycopg2
import psycopg2.extras
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.environ.get(
    "DATABASE_URL",
    f"postgresql://{os.environ.get('USER', 'postgres')}@localhost:5432/conductorv2"
)

HEARTBEAT_TIMEOUT_SECONDS = 10


def get_db():
    conn = psycopg2.connect(DATABASE_URL)
    conn.autocommit = False
    return conn


def serialize(obj):
    if hasattr(obj, 'isoformat'):
        return obj.isoformat()
    return str(obj)


# ── Connection registries ─────────────────────────────────────────
# Two separate registries: dashboard browsers and worker processes.
# They use the same WebSocket transport but carry different message types.
dashboard_clients: list[WebSocket] = []
worker_connections: dict[str, WebSocket] = {}   # worker_id → websocket


async def notify_idle_workers():
    """
    Push "task_available" to all currently-idle worker connections.
    Workers that receive this immediately attempt a claim rather than
    waiting for their next poll cycle — eliminates the up-to-2s
    dead time from v1's polling approach.
    """
    if not worker_connections:
        return
    msg = json.dumps({"type": "task_available"})
    for worker_id, ws in list(worker_connections.items()):
        try:
            await ws.send_text(msg)
        except Exception:
            worker_connections.pop(worker_id, None)


async def push_dashboard_state():
    """Push current queue state to all connected dashboard browsers."""
    if not dashboard_clients:
        return
    try:
        conn = get_db()
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

        cur.execute("""
            SELECT id, name, status, priority, retry_count, max_retries,
                   required_gpus, required_mem_gb,
                   worker_id, created_at, updated_at, error_message
            FROM tasks ORDER BY updated_at DESC LIMIT 30
        """)
        tasks = [dict(t) for t in cur.fetchall()]

        cur.execute("""
            SELECT * FROM workers
            WHERE last_seen > NOW() - INTERVAL '60 seconds'
            ORDER BY last_seen DESC LIMIT 10
        """)
        workers = [dict(w) for w in cur.fetchall()]

        cur.execute("SELECT status, COUNT(*) as count FROM tasks GROUP BY status")
        stats = {row['status']: row['count'] for row in cur.fetchall()}

        cur.close()
        conn.close()

        payload = json.dumps({
            "type": "state_update",
            "tasks": tasks,
            "workers": workers,
            "stats": stats,
        }, default=serialize)

        for client in dashboard_clients[:]:
            try:
                await client.send_text(payload)
            except Exception:
                if client in dashboard_clients:
                    dashboard_clients.remove(client)
    except Exception as e:
        print(f"[dashboard] broadcast error: {e}")


async def failure_detection_loop():
    while True:
        await asyncio.sleep(5)
        try:
            conn = get_db()
            cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            threshold = datetime.now(timezone.utc) - timedelta(seconds=HEARTBEAT_TIMEOUT_SECONDS)

            cur.execute("""
                SELECT id, name, worker_id, retry_count, max_retries,
                       required_gpus, required_mem_gb
                FROM tasks
                WHERE status IN ('claimed', 'running')
                AND last_heartbeat < %s
            """, (threshold,))
            stale = cur.fetchall()

            requeued = False
            for task in stale:
                if task['retry_count'] < task['max_retries']:
                    cur.execute("""
                        UPDATE tasks SET
                            status = 'pending', worker_id = NULL,
                            last_heartbeat = NULL, claimed_at = NULL,
                            retry_count = retry_count + 1,
                            error_message = 'Worker heartbeat timeout — requeued'
                        WHERE id = %s
                    """, (task['id'],))
                    print(f"[monitor] Task {task['id']} ({task['name']}) requeued "
                          f"(retry {task['retry_count']+1}/{task['max_retries']})")
                    requeued = True
                else:
                    cur.execute("""
                        UPDATE tasks SET status = 'failed',
                            error_message = 'Heartbeat timeout — exhausted retries'
                        WHERE id = %s
                    """, (task['id'],))
                    print(f"[monitor] Task {task['id']} permanently failed")

                # Restore the dead worker's resources
                cur.execute("""
                    UPDATE workers SET
                        status = 'dead',
                        available_gpus = total_gpus,
                        available_mem_gb = total_mem_gb,
                        current_task_id = NULL
                    WHERE worker_id = %s
                """, (task['worker_id'],))
                worker_connections.pop(task['worker_id'], None)

                cur.execute("""
                    UPDATE attempts SET status = 'abandoned', finished_at = NOW()
                    WHERE task_id = %s AND worker_id = %s AND status = 'running'
                """, (task['id'], task['worker_id']))

            conn.commit()
            cur.close()
            conn.close()

            if requeued:
                await notify_idle_workers()
                await push_dashboard_state()

        except Exception as e:
            print(f"[monitor] error: {e}")


async def dashboard_broadcast_loop():
    while True:
        await asyncio.sleep(2)
        await push_dashboard_state()


@asynccontextmanager
async def lifespan(app: FastAPI):
    asyncio.create_task(failure_detection_loop())
    asyncio.create_task(dashboard_broadcast_loop())
    yield


app = FastAPI(title="Conductor v2 — Resource-Aware Distributed ML Queue", lifespan=lifespan)


# ── Static dashboard ─────────────────────────────────────────────
@app.get("/")
def dashboard():
    return FileResponse("static/index.html")


# ── Dashboard WebSocket ──────────────────────────────────────────
@app.websocket("/ws/dashboard")
async def dashboard_ws(websocket: WebSocket):
    await websocket.accept()
    dashboard_clients.append(websocket)
    await push_dashboard_state()
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        if websocket in dashboard_clients:
            dashboard_clients.remove(websocket)


# ── Worker WebSocket ─────────────────────────────────────────────
# Workers connect here and hold this connection open. The server pushes
# "task_available" when a new task enters the queue; the worker then
# calls POST /tasks/claim as usual. This removes the poll-interval
# latency from v1 while keeping the claim logic in one place.
@app.websocket("/ws/worker/{worker_id}")
async def worker_ws(websocket: WebSocket, worker_id: str):
    await websocket.accept()
    worker_connections[worker_id] = websocket
    print(f"[ws] Worker {worker_id} connected via WebSocket")
    try:
        while True:
            # We don't expect messages from workers on this channel —
            # it's a push-only notification channel from server to worker.
            # receive_text() just keeps the connection alive.
            await websocket.receive_text()
    except WebSocketDisconnect:
        worker_connections.pop(worker_id, None)
        print(f"[ws] Worker {worker_id} WebSocket disconnected")


# ── Request models ───────────────────────────────────────────────
class SubmitTaskRequest(BaseModel):
    name: str
    payload: dict = {}
    priority: int = 5
    max_retries: int = 3
    required_gpus: int = 1
    required_mem_gb: float = 4.0


class ClaimTaskRequest(BaseModel):
    worker_id: str
    hostname: str = "unknown"
    total_gpus: int = 1
    total_mem_gb: float = 8.0
    available_gpus: int = 1
    available_mem_gb: float = 8.0


class HeartbeatRequest(BaseModel):
    worker_id: str
    epochs_completed: int = 0


class CompleteTaskRequest(BaseModel):
    worker_id: str
    success: bool
    result: dict = {}
    error_message: str = ""
    epochs_completed: int = 0


# ── Endpoints ────────────────────────────────────────────────────
@app.post("/tasks")
async def submit_task(req: SubmitTaskRequest):
    conn = get_db()
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    cur.execute("""
        INSERT INTO tasks
            (name, payload, priority, max_retries, required_gpus, required_mem_gb)
        VALUES (%s, %s, %s, %s, %s, %s)
        RETURNING id, name, status, priority, required_gpus, required_mem_gb,
                  max_retries, created_at
    """, (req.name, psycopg2.extras.Json(req.payload),
          req.priority, req.max_retries,
          req.required_gpus, req.required_mem_gb))
    task = cur.fetchone()
    conn.commit()
    cur.close()
    conn.close()

    # Push notification — idle workers will attempt to claim immediately
    await notify_idle_workers()
    await push_dashboard_state()
    return {"task": dict(task)}


@app.post("/tasks/claim")
async def claim_task(req: ClaimTaskRequest):
    conn = get_db()
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

    # Register or update the worker with its resource capacity
    cur.execute("""
        INSERT INTO workers
            (worker_id, hostname, status,
             total_gpus, total_mem_gb,
             available_gpus, available_mem_gb, last_seen)
        VALUES (%s, %s, 'idle', %s, %s, %s, %s, NOW())
        ON CONFLICT (worker_id) DO UPDATE SET
            last_seen = NOW(),
            status = 'idle',
            available_gpus = EXCLUDED.available_gpus,
            available_mem_gb = EXCLUDED.available_mem_gb
    """, (req.worker_id, req.hostname,
          req.total_gpus, req.total_mem_gb,
          req.available_gpus, req.available_mem_gb))

    # ── Resource-aware atomic claim ──────────────────────────────
    # The WHERE clause now filters on BOTH priority AND resource fit.
    # A worker with 1 available GPU will never claim a task needing 2,
    # even if that task has the highest priority in the queue.
    # FOR UPDATE SKIP LOCKED ensures concurrent workers don't race.
    cur.execute("""
        UPDATE tasks SET
            status = 'claimed',
            worker_id = %s,
            claimed_at = NOW(),
            last_heartbeat = NOW()
        WHERE id = (
            SELECT id FROM tasks
            WHERE status = 'pending'
            AND required_gpus     <= %s
            AND required_mem_gb   <= %s
            ORDER BY priority DESC, created_at ASC
            FOR UPDATE SKIP LOCKED
            LIMIT 1
        )
        RETURNING *
    """, (req.worker_id, req.available_gpus, req.available_mem_gb))

    task = cur.fetchone()

    if task is None:
        conn.commit()
        cur.close()
        conn.close()
        return {"task": None, "message": "No tasks available for this worker's resources"}

    # Deduct resources from the worker
    cur.execute("""
        UPDATE workers SET
            status = 'busy',
            current_task_id = %s,
            available_gpus   = available_gpus   - %s,
            available_mem_gb = available_mem_gb - %s
        WHERE worker_id = %s
    """, (task['id'],
          task['required_gpus'], task['required_mem_gb'],
          req.worker_id))

    # Look up checkpoint for retried tasks
    last_epochs = 0
    if task['retry_count'] > 0:
        cur.execute("""
            SELECT epochs_completed FROM attempts
            WHERE task_id = %s
            ORDER BY attempt_number DESC LIMIT 1
        """, (task['id'],))
        row = cur.fetchone()
        if row:
            last_epochs = row['epochs_completed']

    cur.execute("""
        INSERT INTO attempts (task_id, worker_id, attempt_number, status)
        VALUES (%s, %s, %s, 'running')
    """, (task['id'], req.worker_id, task['retry_count'] + 1))

    conn.commit()
    cur.close()
    conn.close()
    await push_dashboard_state()

    task_dict = dict(task)
    task_dict['resume_from_epoch'] = last_epochs
    return {"task": task_dict}


@app.post("/tasks/{task_id}/heartbeat")
async def heartbeat(task_id: int, req: HeartbeatRequest):
    conn = get_db()
    cur = conn.cursor()
    cur.execute("""
        UPDATE tasks SET status = 'running', last_heartbeat = NOW()
        WHERE id = %s AND worker_id = %s AND status IN ('claimed', 'running')
    """, (task_id, req.worker_id))
    if cur.rowcount == 0:
        conn.rollback()
        cur.close()
        conn.close()
        raise HTTPException(404, "Task not found or not owned by this worker")
    cur.execute("""
        UPDATE attempts SET epochs_completed = %s
        WHERE task_id = %s AND worker_id = %s AND status = 'running'
    """, (req.epochs_completed, task_id, req.worker_id))
    conn.commit()
    cur.close()
    conn.close()
    return {"status": "ok"}


@app.post("/tasks/{task_id}/complete")
async def complete_task(task_id: int, req: CompleteTaskRequest):
    conn = get_db()
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    cur.execute("SELECT * FROM tasks WHERE id = %s", (task_id,))
    task = cur.fetchone()
    if task is None:
        raise HTTPException(404, "Task not found")

    if req.success:
        cur.execute("""
            UPDATE tasks SET status = 'completed', result = %s, worker_id = NULL
            WHERE id = %s
        """, (psycopg2.extras.Json(req.result), task_id))
        cur.execute("""
            UPDATE attempts SET status = 'completed', finished_at = NOW(),
                epochs_completed = %s
            WHERE task_id = %s AND worker_id = %s AND status = 'running'
        """, (req.epochs_completed, task_id, req.worker_id))
        new_status = 'completed'
    else:
        if task['retry_count'] < task['max_retries']:
            cur.execute("""
                UPDATE tasks SET status = 'pending', worker_id = NULL,
                    last_heartbeat = NULL, claimed_at = NULL,
                    retry_count = retry_count + 1, error_message = %s
                WHERE id = %s
            """, (req.error_message, task_id))
            new_status = 'pending (requeued)'
            # Notify workers a task is back in queue
            asyncio.create_task(notify_idle_workers())
        else:
            cur.execute("""
                UPDATE tasks SET status = 'failed', error_message = %s
                WHERE id = %s
            """, (req.error_message, task_id))
            new_status = 'failed'

        cur.execute("""
            UPDATE attempts SET status = 'failed', finished_at = NOW(),
                error_message = %s
            WHERE task_id = %s AND worker_id = %s AND status = 'running'
        """, (req.error_message, task_id, req.worker_id))

    # Restore the worker's resources
    cur.execute("""
        UPDATE workers SET
            status = 'idle',
            current_task_id = NULL,
            available_gpus   = available_gpus   + %s,
            available_mem_gb = available_mem_gb + %s
        WHERE worker_id = %s
    """, (task['required_gpus'], task['required_mem_gb'], req.worker_id))

    conn.commit()
    cur.close()
    conn.close()
    await push_dashboard_state()
    return {"task_id": task_id, "new_status": new_status}


@app.get("/tasks")
def list_tasks(status: str = None, limit: int = 30):
    conn = get_db()
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    if status:
        cur.execute("""
            SELECT id, name, status, priority, retry_count, max_retries,
                   required_gpus, required_mem_gb,
                   worker_id, created_at, updated_at, error_message
            FROM tasks WHERE status = %s::task_status
            ORDER BY updated_at DESC LIMIT %s
        """, (status, limit))
    else:
        cur.execute("""
            SELECT id, name, status, priority, retry_count, max_retries,
                   required_gpus, required_mem_gb,
                   worker_id, created_at, updated_at, error_message
            FROM tasks ORDER BY updated_at DESC LIMIT %s
        """, (limit,))
    tasks = [dict(t) for t in cur.fetchall()]
    cur.close()
    conn.close()
    return {"tasks": tasks}


@app.get("/workers")
def list_workers():
    conn = get_db()
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    cur.execute("SELECT * FROM workers ORDER BY last_seen DESC")
    workers = [dict(w) for w in cur.fetchall()]
    cur.close()
    conn.close()
    return {"workers": workers}


@app.get("/stats")
def queue_stats():
    conn = get_db()
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    cur.execute("SELECT status, COUNT(*) as count FROM tasks GROUP BY status")
    status_counts = {row['status']: row['count'] for row in cur.fetchall()}
    cur.execute("""
        SELECT COUNT(*) as count FROM workers
        WHERE status != 'dead'
        AND last_seen > NOW() - INTERVAL '60 seconds'
    """)
    active = cur.fetchone()['count']
    cur.close()
    conn.close()
    return {"tasks": status_counts, "active_workers": active,
            "total_tasks": sum(status_counts.values())}