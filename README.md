# aqshara-video-worker

Minimal Python worker package for the paper-to-video pipeline.

## Development

Install dependencies:

```bash
uv sync
```

Configure environment:

```bash
cp .env.example .env
```

Required variables include `REDIS_URL`, the callback settings, and the R2 credentials.

Run the worker entrypoint with JSON payload from stdin:

```bash
echo '{"video_job_id":"vjob_1","document_id":"doc_1","actor_id":"user_1","attempt":1}' | \
  uv run python -m aqshara_video_worker.run_job
```

Run tests:

```bash
uv run pytest
```
