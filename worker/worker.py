# worker/worker.py — Conductor v2
#
# Key changes from v1:
#   1. Declares GPU/mem resources on startup; only claims tasks that fit.
#   2. Holds a persistent WebSocket to /ws/worker/{worker_id}.
#      Server pushes {"type": "task_available"} on submit/requeue;
#      worker acts immediately instead of waiting POLL_INTERVAL seconds.
#      Poll loop remains as fallback if the WebSocket drops.
#
# Run:
#   python worker/worker.py
#   python worker/worker.py --gpus 2 --mem 16   # declare more resources
#
# Run multiple in separate terminals to see resource-aware scheduling:
#   python worker/worker.py --gpus 1 --mem 8    # terminal A
#   python worker/worker.py --gpus 4 --mem 32   # terminal B

import os
import sys
import time
import random
import signal
import socket
import uuid
import argparse
import threading
import requests

try:
    import websocket   # pip install websocket-client
    WS_AVAILABLE = True
except ImportError:
    WS_AVAILABLE = False
    print("[worker] websocket-client not installed — falling back to poll only")
    print("[worker] Install with: pip install websocket-client")

SERVER_URL = os.environ.get("CONDUCTOR_SERVER_URL", "http://localhost:8002")
WS_URL     = SERVER_URL.replace("http://", "ws://").replace("https://", "wss://")

POLL_INTERVAL_SECONDS   = 5      # fallback poll when WebSocket is idle
HEARTBEAT_INTERVAL_SECONDS = 4
CRASH_PROBABILITY       = 0.05

WORKER_ID = f"worker-{uuid.uuid4().hex[:8]}"
HOSTNAME  = socket.gethostname()

# Parse resource flags
parser = argparse.ArgumentParser(description="Conductor v2 worker")
parser.add_argument("--gpus", type=int,   default=1,  help="Total GPUs this worker has")
parser.add_argument("--mem",  type=float, default=8.0, help="Total memory (GB) this worker has")
args = parser.parse_args()

TOTAL_GPUS   = args.gpus
TOTAL_MEM_GB = args.mem

# Track what's currently in use so we report available correctly on claim
used_gpus   = 0
used_mem_gb = 0.0


def log(msg):
    print(f"[{WORKER_ID}] {msg}", flush=True)


def available_gpus():
    return TOTAL_GPUS - used_gpus


def available_mem():
    return TOTAL_MEM_GB - used_mem_gb


def claim_task():
    try:
        r = requests.post(f"{SERVER_URL}/tasks/claim", json={
            "worker_id":       WORKER_ID,
            "hostname":        HOSTNAME,
            "total_gpus":      TOTAL_GPUS,
            "total_mem_gb":    TOTAL_MEM_GB,
            "available_gpus":  available_gpus(),
            "available_mem_gb": available_mem(),
        })
        return r.json().get("task")
    except requests.RequestException as e:
        log(f"Could not reach server: {e}")
        return None


def send_heartbeat(task_id, epochs_completed):
    try:
        requests.post(f"{SERVER_URL}/tasks/{task_id}/heartbeat", json={
            "worker_id": WORKER_ID,
            "epochs_completed": epochs_completed,
        })
    except requests.RequestException:
        pass


def report_complete(task_id, success, result={}, error_message="", epochs_completed=0):
    try:
        requests.post(f"{SERVER_URL}/tasks/{task_id}/complete", json={
            "worker_id":        WORKER_ID,
            "success":          success,
            "result":           result,
            "error_message":    error_message,
            "epochs_completed": epochs_completed,
        })
    except requests.RequestException as e:
        log(f"Failed to report completion: {e}")


def run_ml_job(task):
    global used_gpus, used_mem_gb

    payload      = task.get("payload", {})
    model_name   = payload.get("model", "unknown")
    total_epochs = payload.get("epochs", 5)
    seconds_per_epoch = payload.get("seconds_per_epoch", 1.0)
    resume_from  = task.get("resume_from_epoch", 0)

    req_gpus = task.get("required_gpus", 1)
    req_mem  = task.get("required_mem_gb", 4.0)

    # Reserve resources locally
    used_gpus   += req_gpus
    used_mem_gb += req_mem

    log(f"Starting '{task['name']}': model={model_name}, "
        f"epochs={total_epochs}, resuming from {resume_from}, "
        f"using {req_gpus} GPU(s) / {req_mem}GB mem")

    epochs_completed    = resume_from
    last_heartbeat_time = time.time()

    try:
        for epoch in range(resume_from + 1, total_epochs + 1):
            time.sleep(seconds_per_epoch)

            if random.random() < CRASH_PROBABILITY:
                error = f"Simulated crash at epoch {epoch}/{total_epochs}"
                log(f"CRASH: {error}")
                return False, {}, error, epochs_completed

            epochs_completed += 1
            log(f"Epoch {epochs_completed}/{total_epochs} complete")

            if time.time() - last_heartbeat_time >= HEARTBEAT_INTERVAL_SECONDS:
                send_heartbeat(task['id'], epochs_completed)
                last_heartbeat_time = time.time()

        result = {
            "model":            model_name,
            "epochs_completed": epochs_completed,
            "final_loss":       round(random.uniform(0.05, 0.3), 4),
            "final_accuracy":   round(random.uniform(0.85, 0.99), 4),
        }
        log(f"Job complete: loss={result['final_loss']}, "
            f"accuracy={result['final_accuracy']}")
        return True, result, "", epochs_completed

    finally:
        # Always release resources, even on crash
        used_gpus   -= req_gpus
        used_mem_gb -= req_mem


# ── WebSocket push thread ─────────────────────────────────────────
# Runs in a background thread. When the server pushes "task_available",
# sets an event that the main loop checks — so the main loop wakes up
# and attempts a claim immediately instead of sleeping until next poll.

task_available_event = threading.Event()


def ws_listener():
    """
    Background thread: holds a WebSocket to /ws/worker/{WORKER_ID}.
    Sets task_available_event whenever the server pushes a notification.
    Reconnects automatically on disconnect.
    """
    if not WS_AVAILABLE:
        return

    ws_endpoint = f"{WS_URL}/ws/worker/{WORKER_ID}"

    while True:
        try:
            log(f"Connecting to server WebSocket: {ws_endpoint}")
            ws = websocket.WebSocketApp(
                ws_endpoint,
                on_message=lambda ws, msg: on_ws_message(msg),
                on_error=lambda ws, err: log(f"WebSocket error: {err}"),
                on_close=lambda ws, c, m: log("WebSocket closed, reconnecting..."),
            )
            ws.run_forever()
        except Exception as e:
            log(f"WebSocket thread error: {e}")
        time.sleep(2)   # wait before reconnecting


def on_ws_message(msg):
    try:
        data = json_parse(msg)
        if data.get("type") == "task_available":
            log("Push received: task available")
            task_available_event.set()
    except Exception:
        pass


def json_parse(s):
    import json
    return json.loads(s)


def main():
    log(f"Starting — {TOTAL_GPUS} GPU(s), {TOTAL_MEM_GB}GB mem — server: {SERVER_URL}")

    def handle_shutdown(sig, frame):
        log("Shutting down.")
        sys.exit(0)

    signal.signal(signal.SIGINT, handle_shutdown)
    signal.signal(signal.SIGTERM, handle_shutdown)

    # Start WebSocket listener in background thread
    if WS_AVAILABLE:
        t = threading.Thread(target=ws_listener, daemon=True)
        t.start()
    else:
        log("Running in poll-only mode (install websocket-client for push support)")

    while True:
        task = claim_task()

        if task is None:
            log(f"No tasks available for {available_gpus()} GPU(s) / "
                f"{available_mem()}GB mem — waiting...")
            # Wait for push notification OR fall back to poll after timeout
            # event.wait(timeout) returns True if event was set, False on timeout
            task_available_event.wait(timeout=POLL_INTERVAL_SECONDS)
            task_available_event.clear()
            continue

        log(f"Claimed task {task['id']}: '{task['name']}' "
            f"(priority={task['priority']}, "
            f"needs {task.get('required_gpus',1)} GPU(s) / "
            f"{task.get('required_mem_gb',4)}GB, "
            f"retry {task['retry_count']}/{task['max_retries']})")

        send_heartbeat(task['id'], 0)

        success, result, error_message, epochs_completed = run_ml_job(task)

        report_complete(
            task_id=task['id'],
            success=success,
            result=result,
            error_message=error_message,
            epochs_completed=epochs_completed,
        )

        if success:
            log(f"Task {task['id']} completed successfully.")
        else:
            log(f"Task {task['id']} failed: {error_message}")


if __name__ == "__main__":
    main()