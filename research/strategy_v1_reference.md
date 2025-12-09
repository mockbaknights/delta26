Delta Vortex ML Strategy v1 (frozen)
====================================

Scope
-----
- Frozen reference for Strategy v1 in the new `delta26` repo.
- Source copied from the prior Delta project (`../Delta/src/features` and `../Delta/src/train.py`).
- No code changes were made; this document captures how v1 works for research reproducibility.

Data and Labeling
-----------------
- Inputs: OHLCV time series with `Open, High, Low, Close, Volume`, plus `timestamp`.
- Labeling: Triple Barrier Method (`labels.py::get_triple_barrier_labels`):
  - Look forward up to 15 bars (`time_horizon=15`).
  - Take profit: +0.6% (`take_profit=0.006`) before stop loss: -0.3% (`stop_loss=0.003`) → `target=1`.
  - If stop loss hits first or neither hits within the window → `target=0`.
  - ATR-scaled option exists (TP=2×ATR, SL=1×ATR) if ATR is available.
- Lookahead handling: final 15 rows are dropped during training prep to avoid leakage from the triple-barrier window.
- Target column: `df['target']` (binary profitable outcome ahead).

Feature Set (from `research/features/*.py`)
-------------------------------------------
- `delta`: Directional delta; uses `up_delta` when non-zero else `down_delta`.
- `up_delta`, `down_delta`: Wick-derived directional pressure (Open/Low vs High/Close relationships).
- `strength`: Normalized delta using conditional rolling std devs for positive/negative deltas (controls scale).
- `bear_rejection_ratio`: Upper-wick size relative to total bar range (bearish rejection).
- `bull_rejection_ratio`: Lower-wick size relative to total bar range (bullish rejection).
- `range_pct`: Bar range normalized by close (from `rejection.py`).
- `stretch_z`: Z-score of close distance to VWAP over 120-bar window (requires Volume).
- `rsi(13)`: Momentum oscillator on Close with length 13.
- Notes: Additional helper columns may be present (e.g., `vwap`, rolling stds); excluded from the feature list.

Model
-----
- Type: XGBoost classifier (`binary:logistic`).
- Output: Probability of profitable outcome; thresholded for binary predictions.
- Feature exclusions during training: OHLCV (`Open, High, Low, Close, Volume`), `timestamp`, `target`, plus any user-supplied excludes.

Training Pipeline (from `research/train_v1.py`)
-----------------------------------------------
- `prepare_training_data(df, drop_lookahead_rows=15, exclude_cols=None)`:
  - Drops final 15 rows to align with triple-barrier lookahead.
  - Excludes OHLCV, `timestamp`, `target`, and any extras passed in.
  - Returns `X` (features) and `y` (`target`), removing NaNs.
- `train_xgboost_model(X_train, y_train, **xgb_params)`:
  - Defaults: `objective=binary:logistic`, `eval_metric=logloss`, `n_estimators=100`, `max_depth=6`, `learning_rate=0.1`, `subsample=0.8`, `colsample_bytree=0.8`.
  - `scale_pos_weight` auto-computed if not provided (handles class imbalance).
- `evaluate_model(model, X_test, y_test, threshold=0.5)`:
  - Returns probabilities, thresholded predictions, classification report, and confusion matrix.
- `create_test_dataframe_with_predictions` and `optimize_threshold` helpers are available for analysis.

End-to-End Flow
---------------
1) Load raw OHLCV → apply Delta Vortex features (`research/features/`).
2) Apply triple-barrier labeling to create `target`.
3) `prepare_training_data` removes leakage rows and excludes OHLCV/timestamp/target columns.
4) Train `XGBClassifier`; evaluate on holdout.
5) Outputs: probabilities (`model.predict_proba`) and binary signals (thresholded, default 0.5).

Prediction & Strategy Usage
---------------------------
- Predictions: `proba = model.predict_proba(X)[:, 1]`; `pred = (proba >= threshold).astype(int)`.
- Strategy uses probabilities for sizing/entry confidence; binary flag indicates predicted profitable move.
- Final 15 bars are not used for training/inference labeling to respect lookahead.

Version
-------
- Tag: Delta Vortex ML Strategy v1 (frozen)
