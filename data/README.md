# Data Pipeline (Massive flatfiles -> Delta Vortex)

## Requirements
- `awscli` configured with profile `massive`
- `boto3`, `pandas`, `pyarrow` (see `requirements.txt`; boto3 added)

## Steps (run from repo root)
1) Download flatfiles (parallel, resumable)  
   `python3 data/download_flatfiles.py`

2) Extract QQQ/SPY rows (auto-detect columns)  
   `python3 data/extract_tickers.py`

3) Resample to multi-timeframe OHLCV  
   `python3 data/resample_ohlcv.py`

## Date range customization
- In `data/download_flatfiles.py`, adjust `START_DATE` / `END_DATE` (YYYY-MM-DD).  
- Safe to re-run; existing files are skipped.

## Parallel downloading
- Uses `ThreadPoolExecutor(max_workers=8)` to pull daily files concurrently.
- Logs: downloaded / skipped / missing (404).

## Outputs
- Raw downloads: `data/raw/flatfiles/{year}/{YYYY-MM-DD}.csv.gz`
- 1-minute parquet:  
  - `data/processed/qqq/qqq_1m.parquet`  
  - `data/processed/spy/spy_1m.parquet`
- Resampled parquet (per symbol): `*_2min.parquet`, `*_3min.parquet`, `*_5min.parquet`, `*_10min.parquet`, `*_30min.parquet`, `*_1H.parquet`, `*_4H.parquet`, `*_1D.parquet`

## Adding new tickers
- Update `TICKERS` in `data/extract_tickers.py` and ensure folder naming follows `data/processed/{ticker.lower()}/`.
- Resampling picks up per-symbol 1m sources automatically if added to `SOURCE_FILES`.

## Runtime & best practices
- Expect ~2–3 minutes for a full year of QQQ+SPY with 8 workers (network dependent).
- Re-run safely; missing days just log and continue.
- Keep disk space in mind: per-symbol 1m parquet can be multiple GB for full-year data.

## Usage for research
- Load `data/processed/qqq/qqq_1m.parquet` or resampled parquets for training.
- Paths are relative to repo root; no hard-coded absolutes.

