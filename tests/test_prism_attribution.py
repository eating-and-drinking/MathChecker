"""Tests for counterfactual specialist attribution."""
from __future__ import annotations

import math

from mathchecker.prism.attribution import (
    AttributionEntry,
    ChannelTrace,
    attribute_channels,
    channel_ranking,
    counterfactual_posterior,
    kl_divergence,
    tv_distance,
)
from mathchecker.prism.infer import PrismEvidence, prism_infer


# ---- math primitives ----

def test_counterfactual_with_uniform_likelihood_is_identity() -> None:
    """Removing a channel that emitted uniform likelihood should leave the
    posterior unchanged."""
    post = [0.1, 0.2, 0.3, 0.4]
    uniform = [1.0, 1.0, 1.0, 1.0]
    cf = counterfactual_posterior(factual_probs=post, channel_likelihood=uniform)
    for a, b in zip(post, cf):
        assert math.isclose(a, b, abs_tol=1e-9)


def test_counterfactual_inverts_a_simple_update() -> None:
    """Forward: pi' = pi * lik / Z. Inverse: pi = pi' / lik (renormalized)."""
    prior = [0.25, 0.25, 0.25, 0.25]
    lik = [0.1, 0.9, 0.1, 0.1]
    unnorm = [p * l for p, l in zip(prior, lik)]
    z = sum(unnorm)
    factual = [u / z for u in unnorm]
    recovered = counterfactual_posterior(factual_probs=factual, channel_likelihood=lik)
    for a, b in zip(prior, recovered):
        assert math.isclose(a, b, abs_tol=1e-9)


def test_kl_divergence_zero_for_identical_distributions() -> None:
    p = [0.25, 0.5, 0.25]
    assert math.isclose(kl_divergence(p, p), 0.0, abs_tol=1e-9)


def test_kl_divergence_positive_when_different() -> None:
    p = [0.9, 0.05, 0.05]
    q = [0.1, 0.45, 0.45]
    assert kl_divergence(p, q) > 0.5


def test_tv_distance_bounds() -> None:
    p = [1.0, 0.0]
    q = [0.0, 1.0]
    assert math.isclose(tv_distance(p, q), 1.0, abs_tol=1e-9)
    assert math.isclose(tv_distance(p, p), 0.0, abs_tol=1e-9)


# ---- attribute_channels semantics ----

def test_attribution_zero_for_uniform_channel() -> None:
    num_steps = 3
    final = [0.1, 0.1, 0.7, 0.1]
    traces = [
        ChannelTrace(step_index=2, channel="uniform_channel", likelihood=(1.0, 1.0, 1.0, 1.0)),
    ]
    attrs = attribute_channels(
        final_probs=final, channel_traces=traces, num_steps=num_steps,
    )
    assert len(attrs) == 1
    assert math.isclose(attrs[0].kl_to_counterfactual, 0.0, abs_tol=1e-9)
    assert math.isclose(attrs[0].tv_to_counterfactual, 0.0, abs_tol=1e-9)


def test_attribution_positive_for_concentrated_channel() -> None:
    """A channel whose likelihood was concentrated on the eventually-chosen
    argmax should get a strictly positive attribution and meaningful TV."""
    num_steps = 3
    final = [0.02, 0.02, 0.94, 0.02]
    traces = [
        ChannelTrace(step_index=2, channel="specialist_A", likelihood=(0.05, 0.05, 0.90, 0.05)),
    ]
    attrs = attribute_channels(
        final_probs=final, channel_traces=traces, num_steps=num_steps,
    )
    assert attrs[0].kl_to_counterfactual > 0.3
    assert attrs[0].tv_to_counterfactual > 0.3
    assert attrs[0].factual_argmax == 2


def test_attribution_counterfactual_argmax_shifts_when_only_channel_removed() -> None:
    """If a SINGLE strong channel is the reason the posterior favors k, then
    removing it must shift the counterfactual argmax away from k. We construct
    a posterior derived by applying a single sharp specialist update to a
    uniform prior."""
    num_steps = 3
    prior = [0.25, 0.25, 0.25, 0.25]
    lik = (0.05, 0.05, 0.95, 0.05)
    unnorm = [p * l for p, l in zip(prior, lik)]
    z = sum(unnorm)
    final = [u / z for u in unnorm]
    traces = [ChannelTrace(step_index=2, channel="solo_specialist", likelihood=lik)]
    attrs = attribute_channels(
        final_probs=final, channel_traces=traces, num_steps=num_steps,
    )
    assert attrs[0].factual_argmax == 2
    # Counterfactual recovers the uniform prior -> any of the four hypotheses
    # could win argmax (numerically the first one in tie); definitely not 2
    # being uniquely dominant any more.
    cf = counterfactual_posterior(factual_probs=final, channel_likelihood=lik)
    assert max(cf) - min(cf) < 1e-6  # uniform up to FP error


def test_attribution_orders_consistent_with_likelihood_strength() -> None:
    """A channel that pulled HARDER toward the MAP gets a higher KL than a
    weak corroborating channel."""
    num_steps = 3
    final = [0.02, 0.02, 0.94, 0.02]
    traces = [
        ChannelTrace(step_index=2, channel="strong", likelihood=(0.02, 0.02, 0.94, 0.02)),
        ChannelTrace(step_index=2, channel="weak", likelihood=(0.20, 0.20, 0.40, 0.20)),
    ]
    attrs = attribute_channels(
        final_probs=final, channel_traces=traces, num_steps=num_steps,
    )
    ranking = channel_ranking(attrs, by="kl")
    assert ranking[0].channel == "strong"
    assert ranking[1].channel == "weak"


def test_channel_ranking_rejects_invalid_by() -> None:
    try:
        channel_ranking([], by="garbage")
    except ValueError:
        return
    raise AssertionError("invalid `by` should raise")


# ---- end-to-end wiring through prism_infer ----

def test_prism_infer_attribution_off_returns_empty_list() -> None:
    num_steps = 3

    def ev(t):
        return PrismEvidence(
            step_index=t,
            principle_labels=("correct-and-aligned",) * 3,
        )

    result = prism_infer(num_steps=num_steps, evidence_at_step=ev)
    assert result.attribution == []


def test_prism_infer_attribution_on_records_channels() -> None:
    """With attribution=True, each likelihood update should produce one
    AttributionEntry. For a trace where step 2 fires stage1 + stage2 + 1
    specialist, that step contributes 3 entries; other steps contribute 1
    (stage2 only) each. With early-stop, the loop may terminate before
    visiting all steps."""
    num_steps = 4

    def ev(t):
        if t == 2:
            return PrismEvidence(
                step_index=t,
                principle_labels=("contradiction-found",) * 3,
                specialist_emissions={"alternative_route_verifier_tool": (0.92, 0.05)},
                stage1_inconsistency=0.8,
            )
        return PrismEvidence(
            step_index=t,
            principle_labels=("correct-and-aligned",) * 3,
        )

    result = prism_infer(
        num_steps=num_steps,
        evidence_at_step=ev,
        early_stop=False,
        attribution=True,
    )
    # Per step: at least stage2; at step 2 also stage1 + specialist.
    channels_per_step = {t: [] for t in range(num_steps)}
    for entry in result.attribution:
        channels_per_step[entry.step_index].append(entry.channel)

    # Step 2 should have stage1, stage2, and at least one specialist tool name.
    s2_chs = channels_per_step[2]
    assert "stage1" in s2_chs
    assert "stage2" in s2_chs
    assert any(c.endswith("_tool") for c in s2_chs)


def test_prism_infer_attribution_specialist_has_positive_kl_when_dominant() -> None:
    """When stage2 is ambiguous (correct-and-aligned at step 2) but a
    specialist fires hard, the specialist's KL attribution should dominate
    the stage2 entry at the same step. We don't require a specific absolute
    threshold; only that the specialist's KL exceeds stage2's."""
    num_steps = 4

    def ev(t):
        if t == 2:
            return PrismEvidence(
                step_index=t,
                # ambiguous labels: stage2 stays near-uniform
                principle_labels=("correct-and-aligned", "nothing-extracted", "correct-and-aligned"),
                specialist_emissions={"alternative_route_verifier_tool": (0.95, 0.02)},
            )
        return PrismEvidence(
            step_index=t,
            principle_labels=("correct-and-aligned",) * 3,
        )

    result = prism_infer(
        num_steps=num_steps,
        evidence_at_step=ev,
        early_stop=False,
        attribution=True,
    )
    spec_at_2 = next(
        (e for e in result.attribution
         if e.channel == "alternative_route_verifier_tool" and e.step_index == 2),
        None,
    )
    stage2_at_2 = next(
        (e for e in result.attribution
         if e.channel == "stage2" and e.step_index == 2),
        None,
    )
    assert spec_at_2 is not None and stage2_at_2 is not None
    assert spec_at_2.kl_to_counterfactual > 0.0
    assert spec_at_2.kl_to_counterfactual > stage2_at_2.kl_to_counterfactual, (
        f"specialist KL {spec_at_2.kl_to_counterfactual:.4f} did not exceed "
        f"stage2 KL {stage2_at_2.kl_to_counterfactual:.4f}"
    )


def test_attribution_serialization_round_trip() -> None:
    """PrismResult.to_dict should expose attribution entries with the agreed
    field names so downstream tooling can deserialize without a custom path."""
    num_steps = 3

    def ev(t):
        if t == 1:
            return PrismEvidence(
                step_index=t,
                principle_labels=("contradiction-found",) * 3,
                specialist_emissions={"equivalence_substitution_verifier_tool": (0.85, 0.05)},
            )
        return PrismEvidence(
            step_index=t,
            principle_labels=("correct-and-aligned",) * 3,
        )

    result = prism_infer(
        num_steps=num_steps,
        evidence_at_step=ev,
        early_stop=False,
        attribution=True,
    )
    payload = result.to_dict()
    assert "attribution" in payload
    for entry in payload["attribution"]:
        for required in ("step_index", "channel", "kl", "tv", "factual_argmax", "counterfactual_argmax"):
            assert required in entry
 2),
        None,
    )
    assert spec_at_2 is not None and stage2_at_2 is not None
    assert spec_at_2.kl_to_counterfactual > 0.0
    assert spec_at_2.kl_to_counterfactual > stage2_at_2.kl_to_counterfactual, (
        f"specialist KL {spec_at_2.kl_to_counterfactual:.4f} did not exceed "
        f"stage2 KL {stage2_at_2.kl_to_counterfactual:.4f}"
    )


def test_attribution_serialization_round_trip() -> None:
    """PrismResult.to_dict should expose attribution entries with the agreed
    field names so downstream tooling can deserialize without a custom path."""
    num_steps = 3

    def ev(t):
        if t == 1:
            return PrismEvidence(
                step_index=t,
                principle_labels=("contradiction-found",) * 3,
                specialist_emissions={"equivalence_substitution_verifier_tool": (0.85, 0.05)},
            )
        return PrismEvidence(
            step_index=t,
            principle_labels=("correct-and-aligned",) * 3,
        )

    result = prism_infer(
        num_steps=num_steps,
        evidence_at_step=ev,
        early_stop=False,
        attribution=True,
    )
    payload = result.to_dict()
    assert "attribution" in payload
    for entry in payload["attribution"]:
        for required in ("step_index", "channel", "kl", "tv", "factual_argmax", "counterfactual_argmax"):
            assert required in entry
