# Conductor v2 — Resource-Aware Distributed ML Task Queue

An evolution of [Conductor v1](https://github.com/harshavardhanreddyseethagari-sjsu2103/conductor),
adding two HPC-critical features:

1. **Resource-aware scheduling** — tasks declare GPU and memory requirements;
   workers declare their capacity; the claim query only matches tasks that
   fit the worker's available resources. A 1-GPU worker never blocks on or
   steals a 4-GPU job.

2. **Push instead of poll** — workers hold a persistent WebSocket to the
   server. When a task is submitted or requeued, the server immediately
   pushes `{"type": "task_available"}` to all idle workers — eliminating
   the up-to-N-second polling dead time from v1. The poll loop remains
   as a fallback.

Everything from v1 is preserved: atomic claiming via `FOR UPDATE SKIP LOCKED`,
heartbeat-based failure detection, checkpoint-based retry resumption,
configurable per-task retry policy, and a live WebSocket dashboard.

## Architecture

```
Producer (curl / UI)
        │ POST /tasks
        ▼
  Queue Server ──── Postgres (tasks, workers, attempts)
  (FastAPI)    ──── WebSocket /ws/dashboard → browser
        │
        │ push: {"type": "task_available"}
        ▼
  Worker A  Worker B  Worker C   (separate processes)
  1 GPU     2 GPU     4 GPU      each declares its own capacity
  8GB mem   16GB mem  32GB mem
```

## What's different from Conductor v1

| Feature | v1 | v2 |
|---|---|---|
| Resource-aware claiming | ✗ | ✓ tasks declare GPU/mem; workers filter on fit |
| Server→worker push | ✗ poll every 2s | ✓ WebSocket push on submit/requeue |
| Worker resource tracking | ✗ | ✓ available_gpus/mem updated on claim/release |
| Dashboard resource display | ✗ | ✓ per-task requirements + per-worker utilization bars |

## Quickstart

### Prerequisites

- Python 3.10+
- Postgres (`brew install postgresql@16` on Mac)

### Setup

```bash
git clone https://github.com/harshavardhanreddyseethagari-sjsu2103/conductorv2
cd conductorv2

pip install -r requirements.txt

# Create database and apply schema
psql -U $(whoami) -d postgres -c "CREATE DATABASE conductorv2;"
psql -U $(whoami) -d conductorv2 -f database/schema.sql
```

### Configure

Create a `.env` file:
```
DATABASE_URL=postgresql://YOUR_USERNAME@localhost:5432/conductorv2
```

### Run

```bash
# Terminal 1: queue server + dashboard
uvicorn server.server:app --reload --port 8001

# Terminal 2: a small worker (1 GPU, 8GB)
python worker/worker.py --gpus 1 --mem 8

# Terminal 3: a large worker (4 GPU, 32GB)
python worker/worker.py --gpus 4 --mem 32
```

Visit `http://localhost:8001/` for the live dashboard.

### Submit jobs with different resource requirements

```bash
# Small job — any worker can run this
curl -X POST http://localhost:8001/tasks \
  -H "Content-Type: application/json" \
  -d '{
    "name": "small-job",
    "payload": {"model": "resnet50", "epochs": 5, "seconds_per_epoch": 1},
    "priority": 5,
    "required_gpus": 1,
    "required_mem_gb": 4
  }'

# Large job — only the 4-GPU worker can claim this
curl -X POST http://localhost:8001/tasks \
  -H "Content-Type: application/json" \
  -d '{
    "name": "large-job",
    "payload": {"model": "transformer", "epochs": 5, "seconds_per_epoch": 1},
    "priority": 8,
    "required_gpus": 4,
    "required_mem_gb": 16
  }'
```

The 1-GPU worker will skip `large-job` even though it has higher priority —
it doesn't meet the resource requirements. The 4-GPU worker claims it immediately
via WebSocket push notification.

## Testing failure recovery

```bash
# Submit a long job
curl -X POST http://localhost:8001/tasks \
  -H "Content-Type: application/json" \
  -d '{"name": "long-job", "payload": {"model": "bert", "epochs": 20, "seconds_per_epoch": 2}, "required_gpus": 1, "required_mem_gb": 4}'

# Start a worker, wait a few epochs, kill it with Ctrl+C
python worker/worker.py --gpus 1 --mem 8

# Server detects missing heartbeat within ~15 seconds and requeues:
# [monitor] Task N (long-job) requeued after heartbeat timeout (retry 1/3)

# Start a new worker — it resumes from the last completed epoch
python worker/worker.py --gpus 1 --mem 8
```

## Running with Docker

```bash
docker build -t conductorv2 .
docker run -p 8001:8000 --env-file .env conductorv2
```

Workers can be pointed at any server URL:
```bash
CONDUCTOR_SERVER_URL=https://your-server.onrender.com python worker/worker.py --gpus 2 --mem 16
```

## Project structure

```
conductorv2/
├── database/
│   └── schema.sql        # Postgres schema with resource columns
├── server/
│   └── server.py         # FastAPI server — resource-aware claim + WebSocket push
├── worker/
│   └── worker.py         # Worker — declares resources, WebSocket listener
├── static/
│   └── index.html        # Live dashboard
├── .env.example
├── Dockerfile
└── requirements.txt
```

## How this differs from Airflow / Celery / Kafka

- **Airflow / Prefect** — DAG workflow orchestrators for dependent pipeline steps.
  Conductor is a flat task queue; each task is independent.
- **Celery** — similar task queue but uses Redis/RabbitMQ as the broker.
  Conductor uses Postgres directly, trading throughput for full ACID durability
  and inspectability.
- **Kafka** — distributed event log for streaming to many consumers.
  Conductor distributes each task to exactly one worker, then removes it.

Conductor is intentionally simpler than all three — it implements the
foundational primitives (atomic claiming, heartbeat detection, resource
matching, push notification) from scratch to deeply understand the mechanics.

## Honest limitations

- **Single server** — if the server restarts, in-flight tasks wait for
  heartbeat timeout before recovery. Production fix: multiple server instances
  with Redis pub/sub for the push layer.
- **Simulated ML jobs** — workers use `time.sleep()` as the training loop;
  real use would replace `run_ml_job()` with actual PyTorch training code.
- **At-least-once delivery** — a task could execute twice if a worker completes
  but crashes before reporting back. The `attempts` table records all executions
  for idempotency auditing.

## License

MIT