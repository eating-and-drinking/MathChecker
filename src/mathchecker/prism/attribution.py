"""Counterfactual attribution of evidence channels for PRISM.

For each likelihood update applied during inference, we compute the
posterior we WOULD have obtained had we omitted that channel. The
discrepancy between the factual and counterfactual posteriors measures the
local contribution of that channel.

Local-attribution caveat
------------------------
We invert a single likelihood update from the FINAL posterior by dividing
out its likelihood vector and renormalizing. This is exact under the
fixed-trajectory assumption: "all other channels would have fired the
same way had this one been absent". In a fully causal counterfactual, the
downstream EIG router and specialist-invocation set could have differed,
so the true causal effect can be larger or smaller. The local attribution
is the standard primitive used by Shapley-value, LIME, and ablation-style
explanation tools, and is computable in O(channels * hypotheses) time
without re-running inference.

Mathematical statement
----------------------
Let pi^F = pi_T be the factual final posterior and L_c(tau) the likelihood
vector that channel c contributed at some step. Define

    pi^{-c}(tau)  ~  pi^F(tau) / L_c(tau)            (normalized).

Then the local attribution of channel c is

    attr_c = KL(pi^F || pi^{-c})                     (positive when L_c
                                                      moved mass).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Sequence


# ---- channel trace + attribution entry ----

@dataclass(slots=True, frozen=True)
class ChannelTrace:
    """A single likelihood update applied during inference.

    step_index : int
        Step at which the update was applied.
    channel : str
        Channel name. Examples: "stage1", "stage2",
        "alternative_route_verifier_tool", etc.
    likelihood : tuple[float, ...]
        Length-(T+1) likelihood vector that was multiplied into the posterior.
        These are POST-tempering values when a TemperatureMixer is in use.
    """

    step_index: int
    channel: str
    likelihood: tuple[float, ...]


@dataclass(slots=True, frozen=True)
class AttributionEntry:
    step_index: int
    channel: str
    kl_to_counterfactual: float
    tv_to_counterfactual: float
    factual_argmax: int                # -1 for tau=infty
    counterfactual_argmax: int         # -1 for tau=infty


# ---- core math helpers ----

def counterfactual_posterior(
    *,
    factual_probs: Sequence[float],
    channel_likelihood: Sequence[float],
    eps: float = 1e-12,
) -> list[float]:
    """Remove one channel's contribution from `factual_probs`.

    Returns a renormalized vector pi^{-c} = pi^F / L_c (component-wise),
    with `eps` floor on the divisor to avoid blow-up when L_c was effectively
    zero on some hypothesis.
    """
    if len(factual_probs) != len(channel_likelihood):
        raise ValueError("factual_probs and channel_likelihood length mismatch")
    unnorm = [p / max(float(lik), eps) for p, lik in zip(factual_probs, channel_likelihood)]
    total = sum(unnorm)
    if total <= 0.0:
        return list(factual_probs)
    return [u / total for u in unnorm]


def kl_divergence(p: Sequence[float], q: Sequence[float], *, eps: float = 1e-12) -> float:
    """KL(p || q) in nats."""
    if len(p) != len(q):
        raise ValueError("p and q length mismatch")
    kl = 0.0
    for pi, qi in zip(p, q):
        if pi > eps:
            kl += float(pi) * math.log(float(pi) / max(float(qi), eps))
    return max(0.0, kl)


def tv_distance(p: Sequence[float], q: Sequence[float]) -> float:
    """Total variation distance: 0.5 * sum|p_i - q_i|."""
    if len(p) != len(q):
        raise ValueError("p and q length mismatch")
    return 0.5 * sum(abs(float(a) - float(b)) for a, b in zip(p, q))


def _argmax_index(probs: Sequence[float], *, num_steps: int) -> int:
    """Return MAP index in PRISM convention: -1 for tau=infty (last bucket)."""
    if not probs:
        return -1
    best_idx = 0
    best_val = probs[0]
    for i in range(1, len(probs)):
        if probs[i] > best_val:
            best_val = probs[i]
            best_idx = i
    if best_idx == num_steps:
        return -1
    return best_idx


# ---- public attribution API ----

def attribute_channels(
    *,
    final_probs: Sequence[float],
    channel_traces: Sequence[ChannelTrace],
    num_steps: int,
) -> list[AttributionEntry]:
    """Per-channel KL/TV attribution for every update in `channel_traces`.

    Returns one AttributionEntry per ChannelTrace, sorted by step_index then
    channel (stable order matching the input).
    """
    factual_argmax = _argmax_index(final_probs, num_steps=num_steps)
    out: list[AttributionEntry] = []
    for trace in channel_traces:
        cf = counterfactual_posterior(
            factual_probs=final_probs,
            channel_likelihood=trace.likelihood,
        )
        out.append(
            AttributionEntry(
                step_index=trace.step_index,
                channel=trace.channel,
                kl_to_counterfactual=kl_divergence(final_probs, cf),
                tv_to_counterfactual=tv_distance(final_probs, cf),
                factual_argmax=factual_argmax,
                counterfactual_argmax=_argmax_index(cf, num_steps=num_steps),
            )
        )
    return out


def channel_ranking(
    attributions: Sequence[AttributionEntry],
    *,
    by: str = "kl",
) -> list[AttributionEntry]:
    """Sort attributions in descending order by attribution magnitude.

    by : "kl" or "tv"
    """
    if by not in ("kl", "tv"):
        raise ValueError("by must be 'kl' or 'tv'")
    key = (lambda e: e.kl_to_counterfactual) if by == "kl" else (lambda e: e.tv_to_counterfactual)
    return sorted(attributions, key=key, reverse=True)
