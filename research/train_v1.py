"""
Model training for Delta Vortex trading system.

Phase 3: Train XGBoost classifier to predict profitable moves.
"""

from __future__ import annotations

import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from xgboost import XGBClassifier


def prepare_training_data(
    df: pd.DataFrame,
    drop_lookahead_rows: int = 15,
    exclude_cols: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Prepare features (X) and target (y) from labeled DataFrame.
    
    Args:
        df: Labeled DataFrame with features and target column
        drop_lookahead_rows: Number of final rows to drop (default: 15)
        exclude_cols: Additional columns to exclude from features
        
    Returns:
        Tuple of (X_features, y_target)
    """
    df = df.copy()
    
    # Drop the final rows consumed by Triple Barrier lookahead
    if drop_lookahead_rows > 0:
        df = df.iloc[:-drop_lookahead_rows].copy()
    
    # Default columns to exclude from features
    default_exclude = ['target', 'timestamp', 'Open', 'High', 'Low', 'Close', 'Volume']
    if exclude_cols:
        default_exclude.extend(exclude_cols)
    
    # Remove duplicates while preserving order
    exclude_set = set(default_exclude)
    
    # Get feature columns (everything except excluded)
    feature_cols = [col for col in df.columns if col not in exclude_set]
    
    # Ensure target exists
    if 'target' not in df.columns:
        raise ValueError("DataFrame must contain 'target' column")
    
    # Handle NaN targets (shouldn't exist after dropping lookahead, but be safe)
    df = df.dropna(subset=['target'])
    
    X = df[feature_cols].copy()
    y = df['target'].copy()
    
    # Remove any remaining NaN values
    mask = ~(X.isna().any(axis=1) | y.isna())
    X = X[mask]
    y = y[mask]
    
    return X, y


def train_xgboost_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    scale_pos_weight: float | None = None,
    **xgboost_params,
) -> XGBClassifier:
    """
    Train XGBoost classifier with class imbalance handling.
    
    Args:
        X_train: Training features
        y_train: Training target
        scale_pos_weight: Weight for positive class (auto-calculated if None)
        **xgboost_params: Additional XGBoost parameters
        
    Returns:
        Trained XGBoost classifier
    """
    # Auto-calculate scale_pos_weight if not provided
    if scale_pos_weight is None:
        negative_count = (y_train == 0).sum()
        positive_count = (y_train == 1).sum()
        if positive_count > 0:
            scale_pos_weight = negative_count / positive_count
        else:
            scale_pos_weight = 1.0
    
    # Default XGBoost parameters
    default_params = {
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'scale_pos_weight': scale_pos_weight,
        'random_state': 42,
        'n_estimators': 100,
        'max_depth': 6,
        'learning_rate': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
    }
    
    # Override with any provided parameters
    default_params.update(xgboost_params)
    
    # Initialize and train model
    model = XGBClassifier(**default_params)
    model.fit(X_train, y_train)
    
    return model


def evaluate_model(
    model: XGBClassifier,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    threshold: float = 0.5,
) -> dict:
    """
    Evaluate model performance and return predictions with probabilities.
    
    Args:
        model: Trained XGBoost classifier
        X_test: Test features
        y_test: Test target
        threshold: Probability threshold for binary classification
        
    Returns:
        Dictionary with predictions, probabilities, and metrics
    """
    # Get probabilities
    y_proba = model.predict_proba(X_test)[:, 1]
    
    # Binary predictions using threshold
    y_pred = (y_proba >= threshold).astype(int)
    
    # Classification report
    report = classification_report(y_test, y_pred, output_dict=True)
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    return {
        'predictions': y_pred,
        'probabilities': y_proba,
        'classification_report': report,
        'confusion_matrix': cm,
    }


def create_test_dataframe_with_predictions(
    test_df: pd.DataFrame,
    y_proba: np.ndarray,
    y_pred: np.ndarray | None = None,
) -> pd.DataFrame:
    """
    Add model predictions to test DataFrame.
    
    Args:
        test_df: Original test DataFrame
        y_proba: Predicted probabilities
        y_pred: Binary predictions (optional)
        
    Returns:
        DataFrame with model_proba and optionally model_pred columns
    """
    result_df = test_df.copy()
    result_df['model_proba'] = y_proba
    
    if y_pred is not None:
        result_df['model_pred'] = y_pred
    
    return result_df


def optimize_threshold(
    y_true: pd.Series | np.ndarray,
    y_proba: np.ndarray,
    threshold_range: tuple[float, float] = (0.1, 0.9),
    step: float = 0.01,
) -> dict:
    """
    Find optimal threshold that maximizes F1-score.
    
    Args:
        y_true: True target values
        y_proba: Predicted probabilities
        threshold_range: (min, max) threshold range to search
        step: Step size for threshold iteration
        
    Returns:
        Dictionary with optimal threshold, metrics, and full results
    """
    from sklearn.metrics import precision_score, recall_score, f1_score
    
    thresholds = np.arange(threshold_range[0], threshold_range[1] + step, step)
    
    results = []
    best_f1 = -1
    best_threshold = 0.5
    best_metrics = {}
    
    for threshold in thresholds:
        y_pred = (y_proba >= threshold).astype(int)
        
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        
        results.append({
            'threshold': threshold,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
        })
        
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
            best_metrics = {
                'threshold': threshold,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
            }
    
    results_df = pd.DataFrame(results)
    
    return {
        'optimal_threshold': best_threshold,
        'optimal_metrics': best_metrics,
        'all_results': results_df,
    }
