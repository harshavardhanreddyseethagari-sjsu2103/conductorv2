# Conductor — Resource-Aware Distributed ML Task Queue

A distributed task queue for ML training jobs, built to simulate how
HPC schedulers match jobs to compute resources. Workers declare their
GPU and memory capacity; tasks declare what they need; the queue only
assigns a task to a worker that can actually run it.

Workers connect to the server via WebSocket — when a task is submitted,
the server immediately pushes a notification to idle workers so they
claim it without waiting for a poll cycle. If a worker dies mid-job,
the server detects the missing heartbeat and requeues the task
automatically, resuming from the last completed epoch checkpoint.

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
  Worker A    Worker B    Worker C    (separate processes)
  1 GPU       2 GPU       4 GPU       each declares its own capacity
  8GB mem     16GB mem    32GB mem
```

## Core concepts demonstrated

| Concept | Where |
|---|---|
| Resource-aware claiming | Claim query filters `required_gpus <= available_gpus AND required_mem_gb <= available_mem_gb` |
| Atomic claiming | `FOR UPDATE SKIP LOCKED` — concurrent workers never claim the same task |
| Push instead of poll | Workers hold WebSocket to `/ws/worker/{id}`; server pushes on submit/requeue |
| Heartbeat failure detection | Server-side loop requeues tasks whose workers go silent |
| Checkpoint resumption | Retried tasks resume from last epoch stored in `attempts` table |
| Configurable retry policy | `max_retries` set per task, not globally |

## Quickstart

### Prerequisites

- Python 3.10+
- Postgres (`brew install postgresql@16` on Mac)

### Setup

```bash
git clone https://github.com/harshavardhanreddyseethagari-sjsu2103/conductor
cd conductor

pip install -r requirements.txt

psql -U $(whoami) -d postgres -c "CREATE DATABASE conductor;"
psql -U $(whoami) -d conductor -f database/schema.sql
```

### Configure

```bash
# Create a .env file
DATABASE_URL=postgresql://YOUR_USERNAME@localhost:5432/conductor
```

### Run

```bash
# Terminal 1: queue server + dashboard
uvicorn server.server:app --reload --port 8001

# Terminal 2: small worker (1 GPU, 8GB)
python worker/worker.py --gpus 1 --mem 8

# Terminal 3: large worker (4 GPU, 32GB)
python worker/worker.py --gpus 4 --mem 32
```

Visit `http://localhost:8001/` for the live dashboard.

## Submitting jobs

```bash
# Small job — any worker can run this
curl -X POST http://localhost:8001/tasks \
  -H "Content-Type: application/json" \
  -d '{
    "name": "resnet-finetune",
    "payload": {"model": "resnet50", "epochs": 5, "seconds_per_epoch": 1},
    "priority": 5,
    "required_gpus": 1,
    "required_mem_gb": 4
  }'

# Large job — only the 4-GPU worker can claim this
curl -X POST http://localhost:8001/tasks \
  -H "Content-Type: application/json" \
  -d '{
    "name": "transformer-training",
    "payload": {"model": "transformer", "epochs": 5, "seconds_per_epoch": 1},
    "priority": 8,
    "required_gpus": 4,
    "required_mem_gb": 16
  }'
```

The 1-GPU worker skips `transformer-training` even though it has higher
priority — it does not meet the resource requirements. The 4-GPU worker
claims it immediately via WebSocket push.

## Testing failure recovery

```bash
# Submit a long job
curl -X POST http://localhost:8001/tasks \
  -H "Content-Type: application/json" \
  -d '{"name": "long-job", "payload": {"model": "bert", "epochs": 20, "seconds_per_epoch": 2}, "required_gpus": 1, "required_mem_gb": 4}'

# Start a worker, kill it mid-run with Ctrl+C
python worker/worker.py --gpus 1 --mem 8

# Server detects missing heartbeat within ~15 seconds:
# [monitor] Task N (long-job) requeued after heartbeat timeout (retry 1/3)

# New worker resumes from last completed epoch
python worker/worker.py --gpus 1 --mem 8
```

## Docker

```bash
docker build -t conductor .
docker run -p 8001:8000 --env-file .env conductor
```

Workers can target any server URL:
```bash
CONDUCTOR_SERVER_URL=https://your-server.onrender.com \
  python worker/worker.py --gpus 2 --mem 16
```

## Project structure

```
conductor/
├── database/
│   └── schema.sql        # Postgres schema — tasks, workers, attempts
├── server/
│   └── server.py         # FastAPI server — claim API, WebSocket push, failure detection
├── worker/
│   └── worker.py         # Worker process — resource declaration, WebSocket listener
├── static/
│   └── index.html        # Live dashboard
├── .env.example
├── Dockerfile
└── requirements.txt
```

## Limitations

- **Single server** — if the server restarts, in-flight tasks wait for
  heartbeat timeout before recovery. A production setup would run multiple
  server instances with a shared pub/sub layer (e.g. Redis) for push notifications.
- **Simulated ML jobs** — workers use `time.sleep()` as the training loop;
  real use would replace `run_ml_job()` with actual PyTorch training code.
- **At-least-once delivery** — a task could execute twice if a worker completes
  but crashes before reporting back. The `attempts` table records all executions
  for idempotency auditing.

## License

MIT
