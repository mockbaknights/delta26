"use client";

import { useEffect, useMemo, useState } from "react";

type RawSignal = {
  timestamp: string;
  symbol: string;
  interval: string;
  direction: string; // BUY / SELL / FLAT
  confidence_pct: number | null;
  risk_blocked: boolean;
  meta?: {
    reason?: string;
    source?: string;
  };
};

type UiSignal = RawSignal & {
  id: string;
  localTime: string;
};

const STREAM_URL =
  process.env.NEXT_PUBLIC_SIGNAL_STREAM_URL ?? "http://localhost:8001/stream";

function formatLocalTime(iso: string | undefined): string {
  if (!iso) return "—";
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return iso;
  return d.toLocaleTimeString([], {
    hour: "numeric",
    minute: "2-digit",
  });
}

function buildId(s: RawSignal): string {
  return [s.timestamp, s.symbol, s.interval, s.direction, s.confidence_pct].join(
    "|",
  );
}

export default function DashboardPage() {
  const [signals, setSignals] = useState<UiSignal[]>([]);
  const [selectedSymbol, setSelectedSymbol] = useState<string>("ALL");
  const [isConnected, setIsConnected] = useState<boolean>(false);
  const [lastUpdated, setLastUpdated] = useState<string | null>(null);

  const symbols = useMemo(() => {
    const set = new Set<string>();
    for (const s of signals) {
      if (s.symbol) set.add(s.symbol);
    }
    return ["ALL", ...Array.from(set).sort()];
  }, [signals]);

  const filteredSignals = useMemo(
    () =>
      selectedSymbol === "ALL"
        ? signals
        : signals.filter((s) => s.symbol === selectedSymbol),
    [signals, selectedSymbol],
  );

  useEffect(() => {
    const url = STREAM_URL;
    const source = new EventSource(url);

    source.onopen = () => {
      setIsConnected(true);
    };

    source.onerror = () => {
      setIsConnected(false);
    };

    source.onmessage = (event) => {
      try {
        const data: RawSignal = JSON.parse(event.data);

        const normalized: UiSignal = {
          ...data,
          direction: (data.direction || "").toUpperCase(),
          confidence_pct:
            data.confidence_pct != null
              ? Math.max(0, Math.min(100, Math.round(data.confidence_pct)))
              : null,
          id: buildId(data),
          localTime: formatLocalTime(data.timestamp),
        };

        setSignals((prev) => {
          if (prev.length && prev[0].id === normalized.id) return prev;
          const next = [normalized, ...prev];
          return next.slice(0, 200);
        });

        setLastUpdated(formatLocalTime(data.timestamp));
      } catch {
        // ignore malformed lines
      }
    };

    return () => {
      source.close();
    };
  }, []);

  return (
    <div className="min-h-screen bg-gradient-to-b from-black via-[#050711] to-black text-slate-100">
      <div className="pointer-events-none fixed inset-0">
        <div className="absolute -top-32 left-1/2 h-64 w-[40rem] -translate-x-1/2 rounded-full bg-cyan-500/10 blur-3xl" />
        <div className="absolute bottom-[-10rem] right-[-5rem] h-72 w-72 rounded-full bg-indigo-500/20 blur-3xl" />
      </div>

      <main className="relative mx-auto flex max-w-6xl flex-col gap-6 px-4 pb-10 pt-8 md:px-8 md:pt-10">
        <header className="flex flex-col justify-between gap-4 md:flex-row md:items-center">
          <div>
            <h1 className="bg-gradient-to-r from-cyan-400 via-sky-400 to-indigo-400 bg-clip-text text-3xl font-semibold tracking-tight text-transparent md:text-4xl">
              Delta26 · Live Signals
            </h1>
            <p className="mt-2 max-w-xl text-sm text-slate-400 md:text-base">
              Real-time model decisions for SPY, QQQ and friends. Tuned for
              intraday clarity, not noise.
            </p>
          </div>

          <div className="flex flex-wrap items-center gap-3 text-xs md:text-sm">
            <div className="flex items-center gap-2 rounded-full border border-white/10 bg-white/5 px-3 py-1.5 backdrop-blur-xl">
              <span
                className={
                  "h-2 w-2 rounded-full " +
                  (isConnected ? "bg-emerald-400" : "bg-red-500")
                }
              />
              <span className="font-medium">
                {isConnected ? "Stream live" : "Reconnecting…"}
              </span>
            </div>
            <div className="rounded-full border border-white/10 bg-white/5 px-3 py-1.5 backdrop-blur-xl">
              <span className="text-slate-400">Last update:</span>{" "}
              <span className="font-medium">
                {lastUpdated ?? "waiting for first tick"}
              </span>
            </div>
          </div>
        </header>

        <section className="flex flex-wrap items-center justify-between gap-3">
          <div className="inline-flex items-center gap-2 rounded-full border border-white/10 bg-white/5 px-3 py-1.5 text-xs text-slate-300 backdrop-blur-xl md:text-sm">
            <span className="rounded-full bg-emerald-500/10 px-2 py-0.5 text-[10px] font-semibold uppercase tracking-wide text-emerald-300">
              Live
            </span>
            <span>Multi-ticker · Multi-interval · Risk-gated</span>
          </div>

          <div className="flex items-center gap-2">
            <label className="text-xs text-slate-400 md:text-sm">Symbol</label>
            <select
              className="rounded-full border border-white/10 bg-black/40 px-3 py-1.5 text-xs text-slate-100 outline-none ring-0 backdrop-blur-xl md:text-sm"
              value={selectedSymbol}
              onChange={(e) => setSelectedSymbol(e.target.value)}
            >
              {symbols.map((sym) => (
                <option key={sym} value={sym}>
                  {sym}
                </option>
              ))}
            </select>
          </div>
        </section>

        <section className="mt-2 grid gap-3 md:mt-4 md:grid-cols-2 xl:grid-cols-3">
          {filteredSignals.length === 0 && (
            <div className="col-span-full rounded-2xl border border-dashed border-white/10 bg-white/5 px-5 py-8 text-center text-sm text-slate-400 backdrop-blur-xl">
              Waiting for signals from <span className="font-semibold">Delta26</span>…
              <div className="mt-3 text-xs text-slate-500">
                Make sure the backend SSE stream is running at{" "}
                <code className="rounded bg-black/50 px-1 py-0.5 text-[10px]">
                  {STREAM_URL}
                </code>
              </div>
            </div>
          )}

          {filteredSignals.map((s) => {
            const isBuy = s.direction === "BUY";
            const isSell = s.direction === "SELL";
            const isFlat = !isBuy && !isSell;

            const border =
              isBuy && !s.risk_blocked
                ? "border-emerald-400/40"
                : isSell && !s.risk_blocked
                ? "border-rose-400/40"
                : "border-slate-500/40";

            return (
              <article
                key={s.id}
                className={`group relative overflow-hidden rounded-2xl border ${border} bg-white/5 p-4 shadow-[0_18px_60px_rgba(0,0,0,0.65)] backdrop-blur-2xl transition-transform duration-200 hover:-translate-y-0.5`}
              >
                <div
                  className={`pointer-events-none absolute inset-x-0 top-0 h-1 ${
                    isBuy
                      ? "bg-gradient-to-r from-emerald-400 via-emerald-300 to-cyan-400"
                      : isSell
                      ? "bg-gradient-to-r from-rose-400 via-red-300 to-orange-400"
                      : "bg-gradient-to-r from-slate-500 via-slate-400 to-slate-500"
                  }`}
                />

                <div className="mb-3 flex items-start justify-between gap-3">
                  <div>
                    <div className="flex items-baseline gap-2">
                      <span className="text-lg font-semibold tracking-tight md:text-xl">
                        {s.symbol ?? "—"}
                      </span>
                      <span className="rounded-full bg-black/60 px-2 py-0.5 text-[10px] font-medium uppercase tracking-wide text-slate-300">
                        {s.interval ?? "—"}
                      </span>
                    </div>
                    <div className="mt-1 text-xs text-slate-400">{s.localTime}</div>
                  </div>

                  <div className="text-right">
                    <div
                      className={`text-sm font-semibold md:text-base ${
                        isBuy
                          ? "text-emerald-300"
                          : isSell
                          ? "text-rose-300"
                          : "text-slate-300"
                      }`}
                    >
                      {s.direction || "—"}
                    </div>
                    <div className="mt-1 text-[11px] text-slate-400">
                      {s.risk_blocked ? (
                        <span className="rounded-full bg-amber-500/10 px-2 py-0.5 font-medium text-amber-300">
                          Risk-gated
                        </span>
                      ) : (
                        <span className="text-slate-500">
                          {isFlat ? "No signal" : "Live eligible"}
                        </span>
                      )}
                    </div>
                  </div>
                </div>

                <div className="mt-1">
                  <div className="flex items-center justify-between text-[11px] text-slate-400">
                    <span>Confidence</span>
                    <span className="font-medium text-slate-200">
                      {s.confidence_pct != null ? `${s.confidence_pct}%` : "—"}
                    </span>
                  </div>
                  <div className="mt-1 h-1.5 overflow-hidden rounded-full bg-black/50">
                    <div
                      className={`h-full rounded-full ${
                        isBuy
                          ? "bg-gradient-to-r from-emerald-400 via-cyan-400 to-sky-400"
                          : isSell
                          ? "bg-gradient-to-r from-rose-400 via-red-400 to-orange-400"
                          : "bg-gradient-to-r from-slate-400 via-slate-300 to-slate-400"
                      }`}
                      style={{
                        width: `${s.confidence_pct != null ? s.confidence_pct : 0}%`,
                      }}
                    />
                  </div>
                </div>

                {(s.meta?.reason || s.meta?.source) && (
                  <div className="mt-3 flex items-center justify-between gap-2 text-[11px] text-slate-500">
                    <div className="line-clamp-1">{s.meta?.reason ?? ""}</div>
                    {s.meta?.source && (
                      <span className="rounded-full bg-black/60 px-2 py-0.5 text-[10px] uppercase tracking-wide text-slate-400">
                        {s.meta.source}
                      </span>
                    )}
                  </div>
                )}

                <div className="pointer-events-none absolute inset-0 opacity-0 transition-opacity duration-200 group-hover:opacity-60">
                  <div
                    className={`absolute -inset-x-10 -bottom-24 h-40 ${
                      isBuy
                        ? "bg-emerald-500/20"
                        : isSell
                        ? "bg-rose-500/25"
                        : "bg-slate-500/10"
                    } blur-3xl`}
                  />
                </div>
              </article>
            );
          })}
        </section>
      </main>
    </div>
  );
}
