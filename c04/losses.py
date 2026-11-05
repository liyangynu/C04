"""Loss functions used by the noisy-label training routes."""

import torch
import torch.nn.functional as F


def generalized_cross_entropy(logits, targets, q=0.7):
    """Return per-example Generalized Cross Entropy (GCE) losses.

    GCE is ``(1 - p_y**q) / q`` for ``q > 0``.  The implementation uses
    log-softmax so that small probabilities do not underflow before the
    power operation.  Reduction is deliberately left to the caller so the
    CE and GCE training routes share the same batch accounting.
    """
    if not torch.is_floating_point(logits) or logits.ndim != 2:
        raise ValueError("logits must be a floating point matrix")
    if targets.ndim != 1 or targets.shape[0] != logits.shape[0]:
        raise ValueError("targets must contain one class index per example")
    if not isinstance(q, (float, int)) or not 0 < float(q) <= 1:
        raise ValueError("GCE q must lie in (0, 1]")
    log_probability = F.log_softmax(logits, dim=1).gather(1, targets.long().view(-1, 1)).squeeze(1)
    probability_power = torch.exp(float(q) * log_probability)
    return (1.0 - probability_power) / float(q)


def training_loss(logits, targets, config):
    """Compute the configured unreduced loss and validate its parameters."""
    name = config.get("loss", "ce").lower()
    if name == "ce":
        return F.cross_entropy(logits, targets, reduction="none")
    if name == "gce":
        return generalized_cross_entropy(logits, targets, config.get("q", 0.7))
    raise ValueError(f"Unknown training loss: {name}")
