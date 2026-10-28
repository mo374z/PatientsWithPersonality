import re
import statistics

BACKCHANNEL = re.compile(
    r"^(yeah|yep|yes|no|nope|okay|ok|right|uh huh|mm hmm|sure|correct|exactly|alright)[\s.,!?]*$",
    re.I,
)
DISFLUENCY = re.compile(r"\b(uh|um|er|hmm|mm|you know|i mean|like|well)\b", re.I)
CONTRACTION = re.compile(r"\b\w+'(s|t|re|ve|ll|d|m)\b", re.I)

METRICS = (
    "mean_tokens",
    "one_word_rate",
    "backchannel_rate",
    "disfluency_per_turn",
    "contraction_per_turn",
)

# How much the patient says depends on the personality parametrisation (the X axis
# and the answer_form mechanism), so only the personality-invariant register
# markers are comparable to a real transcript across HEXACO configurations.
REGISTER_METRICS = ("disfluency_per_turn", "contraction_per_turn")

FORMS = ("polar", "open", "statement")


def _clean(texts):
    return [str(t).strip() for t in texts if isinstance(t, str) and str(t).strip()]


def style_profile(texts: list[str]) -> dict[str, float]:
    """Surface-form speech statistics for a set of patient turns."""
    texts = _clean(texts)
    if not texts:
        raise ValueError("style_profile requires at least one non-empty turn")
    tokens = [len(t.split()) for t in texts]
    n = len(texts)
    return {
        "mean_tokens": sum(tokens) / n,
        "one_word_rate": sum(1 for t in tokens if t <= 1) / n,
        "backchannel_rate": sum(1 for t in texts if BACKCHANNEL.match(t)) / n,
        "disfluency_per_turn": sum(len(DISFLUENCY.findall(t)) for t in texts) / n,
        "contraction_per_turn": sum(len(CONTRACTION.findall(t)) for t in texts) / n,
    }


def profile_by_form(texts: list[str], forms: list[str]) -> dict[str, dict[str, float]]:
    """Style profile per conversational form, for the bimodality the corpus shows."""
    if len(texts) != len(forms):
        raise ValueError("texts and forms must be the same length")
    grouped = {}
    for form in FORMS:
        subset = [t for t, f in zip(texts, forms) if f == form and str(t).strip()]
        if subset:
            grouped[form] = style_profile(subset)
    return grouped


def corpus_scales(profiles: list[dict[str, float]]) -> dict[str, float]:
    """Per-metric spread across cases, used to make metric deviations comparable."""
    if len(profiles) < 2:
        raise ValueError("corpus_scales needs at least two profiles")
    scales = {}
    for metric in METRICS:
        values = [p[metric] for p in profiles]
        spread = statistics.pstdev(values)
        scales[metric] = spread if spread > 1e-9 else 1.0
    return scales


def style_distance(
    simulated: dict[str, float],
    real: dict[str, float],
    scales: dict[str, float],
    metrics: tuple[str, ...] = METRICS,
) -> float:
    """Mean absolute deviation across metrics, each in units of its corpus spread."""
    return sum(abs(simulated[m] - real[m]) / scales[m] for m in metrics) / len(metrics)


def register_distance(
    simulated: dict[str, float],
    real: dict[str, float],
    scales: dict[str, float],
) -> float:
    """Style distance over the register markers only."""
    return style_distance(simulated, real, scales, metrics=REGISTER_METRICS)


def register_band(profiles: list[dict[str, float]], lower: float, upper: float):
    """Range each register marker spans across a corpus of real speakers."""
    if not 0.0 <= lower < upper <= 100.0:
        raise ValueError(f"invalid percentile band ({lower}, {upper})")
    quantiles = {}
    for metric in REGISTER_METRICS:
        values = sorted(p[metric] for p in profiles)
        quantiles[metric] = (
            _percentile(values, lower),
            _percentile(values, upper),
        )
    return quantiles


def _percentile(sorted_values: list[float], q: float) -> float:
    if not sorted_values:
        raise ValueError("percentile of an empty sequence")
    position = (len(sorted_values) - 1) * q / 100.0
    low = int(position)
    high = min(low + 1, len(sorted_values) - 1)
    return sorted_values[low] + (sorted_values[high] - sorted_values[low]) * (
        position - low
    )


def register_penalty(
    simulated: dict[str, float],
    band: dict[str, tuple[float, float]],
    scales: dict[str, float],
) -> float:
    """How far the register falls outside the range real speakers occupy.

    Zero anywhere inside the band, so once the register is humanly plausible this
    term cannot trade against the personality or information objectives.
    """
    total = 0.0
    for metric in REGISTER_METRICS:
        low, high = band[metric]
        value = simulated[metric]
        excess = max(low - value, value - high, 0.0)
        total += excess / scales[metric]
    return total / len(REGISTER_METRICS)


def style_distance_by_form(
    simulated: dict[str, dict[str, float]],
    real: dict[str, dict[str, float]],
    scales: dict[str, float],
) -> float:
    """Style distance averaged over the forms both sides share."""
    shared = [f for f in FORMS if f in simulated and f in real]
    if not shared:
        raise ValueError("no shared conversational form between simulated and real")
    return sum(
        style_distance(simulated[f], real[f], scales) for f in shared
    ) / len(shared)
