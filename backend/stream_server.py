import asyncio
import json
from pathlib import Path

import aiofiles
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # adjust to restrict to your Vercel domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


async def tail_jsonl(path: Path):
    async with aiofiles.open(path, "r") as f:
        await f.seek(0, 2)  # jump to end
        while True:
            line = await f.readline()
            if not line:
                await asyncio.sleep(0.2)
                continue

            try:
                raw = json.loads(line)
            except json.JSONDecodeError:
                continue

            ts = raw.get("timestamp")
            symbol = raw.get("symbol")
            interval = raw.get("interval")
            signal = (raw.get("signal") or "").upper()
            conf = raw.get("confidence")
            confidence_pct = None
            if conf is not None:
                confidence_pct = int(conf * 100) if conf <= 1 else int(conf)

            payload = {
                "timestamp": ts,
                "symbol": symbol,
                "interval": interval,
                "direction": signal,
                "confidence_pct": confidence_pct,
                "risk_blocked": bool(raw.get("risk_blocked", False)),
                "meta": {
                    "reason": raw.get("reason"),
                    "source": raw.get("source"),
                },
            }

            yield f"data: {json.dumps(payload)}\n\n"


@app.get("/stream")
async def stream():
    jsonl_path = Path(__file__).resolve().parent.parent / "data" / "live_stream.jsonl"
    return StreamingResponse(tail_jsonl(jsonl_path), media_type="text/event-stream")

