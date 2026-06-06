# Evaluation metrics

The evaluation module computes per-phase metrics for train and test data:

- MSE
- RMSE
- MAE
- R²
- MAPE with zero-safe masking
- coefficient of determination
- Huber loss
- quantile loss

The output is a tidy CSV table with columns such as `model`, `split`, `phase`, and metric names.
