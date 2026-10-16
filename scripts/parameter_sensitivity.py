"""Batched one-input sensitivity sweeps; not outcome simulations."""
import numpy as np

def sweep(model, base, feature_index, low, high,
          points=1_000_001, batch_size=8192):
    base = np.asarray(base, dtype=float)
    if base.ndim != 1 or not 0 <= feature_index < base.size:
        raise ValueError("Invalid feature vector or index")
    if not np.isfinite(low) or not np.isfinite(high) or low >= high:
        raise ValueError("Bounds must be finite and increasing")
    if points < 2 or batch_size < 1 or batch_size > 65536:
        raise ValueError("Invalid grid size or batch size")
    minimum, maximum, total, count = 1.0, 0.0, 0.0, 0
    step = (high - low) / (points - 1)
    for start in range(0, points, batch_size):
        stop = min(start + batch_size, points)
        values = low + np.arange(start, stop, dtype=float) * step
        batch = np.tile(base, (stop - start, 1))
        batch[:, feature_index] = values
        probabilities = model.predict_proba(batch)[:, 1]
        if not np.all(np.isfinite(probabilities)):
            raise ValueError("Model returned nonfinite probabilities")
        minimum = min(minimum, float(probabilities.min()))
        maximum = max(maximum, float(probabilities.max()))
        total += float(probabilities.sum())
        count += len(probabilities)
    return {
        "points": count, "low": float(low), "high": float(high),
        "minimum_home_win_probability": minimum,
        "maximum_home_win_probability": maximum,
        "uniform_grid_mean_not_a_forecast": total / count,
        "interpretation": "One-input sensitivity; other inputs held fixed. Bounds are exploratory, not a probability distribution.",
    }
