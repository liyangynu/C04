"""Separate, ID-aligned training weights with a fixed global mean of one."""
import numpy as np
import torch


def validate_weights(rows, weights):
    if not isinstance(weights, dict) or set(weights) != {r['id'] for r in rows}:
        raise ValueError('Weights must contain exactly the training IDs')
    values = np.array([weights[r['id']] for r in rows], dtype=np.float64)
    if not np.isfinite(values).all() or np.any(values <= 0):
        raise ValueError('Weights must be finite and positive')
    if not np.isclose(values.mean(), 1, rtol=0, atol=1e-10):
        raise ValueError('Weights must have global training mean one')
    return {'rows': len(values), 'min': float(values.min()), 'max': float(values.max()),
            'mean': float(values.mean()), 'effective_sample_size': float(values.sum()**2/(values**2).sum()),
            'reduction': 'mean(weight * per_example_loss); weights globally normalized, not per batch'}


def reduce_loss(losses, weights=None):
    if weights is None:
        return losses.mean()
    if weights.shape != losses.shape or not torch.isfinite(weights).all() or torch.any(weights <= 0):
        raise ValueError('Batch weights must be positive and match losses')
    return (losses*weights).mean()
