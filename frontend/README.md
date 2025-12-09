# Delta26 Live Dashboard (Next.js + Tailwind)

Apple / TradingView–style live feed for your Delta26 signals. Consumes the FastAPI SSE stream that tails `data/live_stream.jsonl`.

## Environment

Set the stream URL in `.env.local`:

```bash
NEXT_PUBLIC_SIGNAL_STREAM_URL="http://localhost:8001/stream"
```

In Vercel, add the same env var in the project settings (use your public SSE URL).

## Local run

From repo root:

```bash
# start SSE backend (from repo root)
uvicorn backend.stream_server:app --host 0.0.0.0 --port 8001

# in another shell, run the dashboard
cd frontend
pnpm dev
```

Open http://localhost:3000 to see the live grid. Signals appear as the trading engine appends lines to `data/live_stream.jsonl`.

## Deploy (Vercel)

1. Push this repo to GitHub (or link directly in Vercel).
2. In Vercel project settings, set `NEXT_PUBLIC_SIGNAL_STREAM_URL` to your reachable SSE endpoint (e.g., `https://your-server/stream`).
3. Deploy; the dashboard will connect to the SSE stream at runtime.
