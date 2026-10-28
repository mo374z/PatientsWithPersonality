"""Shared display-name maps for plotting in evaluation notebooks."""

import re

PWP_DISPLAY_NAMES = {
    "PatientsWithPersonality_H2_E1_X1_A2_C2_O1_LB": "PatientsWithPersonality\n(Default)",
    "PatientsWithPersonality_H3_E1_X1_A1_C1_O1_LB": "PatientsWithPersonality\n(Dishonest)",
    "PatientsWithPersonality_H1_E3_X1_A1_C1_O1_LB": "PatientsWithPersonality\n(Emotional)",
    "PatientsWithPersonality_H1_E1_X3_A1_C1_O1_LB": "PatientsWithPersonality\n(Extraverted)",
    "PatientsWithPersonality_H1_E1_X1_A3_C1_O1_LB": "PatientsWithPersonality\n(Frustrated/Skeptical)",
    "PatientsWithPersonality_H1_E1_X1_A1_C3_O1_LB": "PatientsWithPersonality\n(Disorganized)",
    "PatientsWithPersonality_H1_E1_X1_A1_C1_O3_LB": "PatientsWithPersonality\n(Cautious)",
}

PATIENTSIM_DISPLAY_NAMES = {
    "PatientSimPatient_persplain_cefrB_dazednormal_recallhigh": "PatientSim\n(Default)",
    "PatientSimPatient_persdistrust_cefrB_dazednormal_recalllow": "PatientSim\n(Distrust+LowRecall)",
    "PatientSimPatient_persoveranxious_cefrB_dazednormal_recallhigh": "PatientSim\n(Overanxious)",
    "PatientSimPatient_persverbose_cefrB_dazednormal_recallhigh": "PatientSim\n(Verbose)",
    "PatientSimPatient_persimpatient_cefrB_dazednormal_recallhigh": "PatientSim\n(Impatient)",
    "PatientSimPatient_persplain_cefrB_dazedhigh_recalllow": "PatientSim\n(Dazed+LowRecall)",
    "PatientSimPatient_persdistrust_cefrB_dazednormal_recallhigh": "PatientSim\n(Distrust)",
}

OTHER_PATIENT_DISPLAY_NAMES = {
    "direct": "Full Case",
    "CraftMDPatient": "CraftMD",
    "VirtualPatient": "VirtualPatient",
    "StateAwarePatient": "StateAwarePatient",
    "AgentClinicPatient_nobias": "AgentClinic",
    "BaselinePatient": "Human Rephrase",
}

PATIENT_DISPLAY_NAMES = {
    **PWP_DISPLAY_NAMES,
    **PATIENTSIM_DISPLAY_NAMES,
    **OTHER_PATIENT_DISPLAY_NAMES,
}


PWP_DEFAULT_NAME = "PatientsWithPersonality_H2_E1_X1_A2_C2_O1_LB"
PATIENTSIM_DEFAULT_NAME = "PatientSimPatient_persplain_cefrB_dazednormal_recallhigh"

PWP_DEFAULT_GROUP = "PatientsWithPersonality\n(Default)"
PWP_POPULATION_GROUP = "PatientsWithPersonality\n(Population)"
PWP_PROBE_GROUP = "PatientsWithPersonality\n(Single-Axis Probe)"

PWP_PROBE_NAMES = [n for n in PWP_DISPLAY_NAMES if n != PWP_DEFAULT_NAME]

_PWP_AXES = re.compile(
    r"PatientsWithPersonality_H(\d)_E(\d)_X(\d)_A(\d)_C(\d)_O(\d)_L\w"
)

HEXACO_AXES = ["H", "E", "X", "A", "C", "O"]

LEGACY_PWP_PREFIXES = ("PersonaPatient", "EverydayPatient")


def canonical_patient_name(name: str) -> str:
    for prefix in LEGACY_PWP_PREFIXES:
        if name.startswith(prefix):
            return name.replace(prefix, "PatientsWithPersonality", 1)
    return name


def pwp_target_axes(name: str) -> dict[str, int] | None:
    """Configured HEXACO target levels encoded in a PWP persona name."""
    m = _PWP_AXES.match(canonical_patient_name(name))
    return dict(zip(HEXACO_AXES, (int(v) for v in m.groups()))) if m else None


def pwp_persona_label(name: str) -> str:
    """Readable two-line label for any PWP persona (falls back to raw HEXACO code)."""
    m = _PWP_AXES.match(name)
    if not m:
        return name
    h, e, x, a, c, o = m.groups()
    return f"PatientsWithPersonality\n(H{h} E{e} X{x} A{a} C{c} O{o})"


def patient_display_name(name: str) -> str:
    """Per-variant label for any patient, including unnamed sampled cohort personas."""
    if name in PATIENT_DISPLAY_NAMES:
        return PATIENT_DISPLAY_NAMES[name]
    if name.startswith("PatientsWithPersonality"):
        return pwp_persona_label(name)
    return name


def patient_group(name: str) -> str:
    """Collapse personas into aggregation buckets.

    The sampled PWP cohort (any persona not in the curated map) forms the
    representative "(Population)"; the curated single-axis extremes stay in their
    own "(Single-Axis Probe)" bucket so they never bias the population aggregate.
    """
    if name.startswith("PatientsWithPersonality"):
        if name == PWP_DEFAULT_NAME:
            return PWP_DEFAULT_GROUP
        if name in PWP_DISPLAY_NAMES:
            return PWP_PROBE_GROUP
        return PWP_POPULATION_GROUP
    if name.startswith("PatientSimPatient"):
        return (
            "PatientSim\n(Default)"
            if name == PATIENTSIM_DEFAULT_NAME
            else "PatientSim\n(Population)"
        )
    return PATIENT_DISPLAY_NAMES.get(name, name)


def patient_family(name: str) -> str:
    """The patient-simulator family, ignoring parametrization — for shape/marker grouping.

    All PatientsWithPersonality personas collapse to one family, all PatientSim personas
    to another; single-variant simulators map to their own display name.
    """
    if name.startswith("PatientsWithPersonality"):
        return "PatientsWithPersonality"
    if name.startswith("PatientSimPatient"):
        return "PatientSim"
    return PATIENT_DISPLAY_NAMES.get(name, name)


MODEL_DISPLAY_NAMES = {
    "gemini-3.5-flash": "Gemini 3.5 Flash",
    "gemini-3.1-flash-lite": "Gemini 3.1 Flash Lite",
    "gemini-3.1-pro-preview-high-thinking-default": "Gemini 3.1 Pro (Thinking)",
    "qwen-3.6-35b-a3b": "Qwen 3.6 35B",
    "gemma-4": "Gemma 4 31B",
    "openai/gpt-4o": "GPT-4o",
    "claude-sonnet-4-6": "Claude Sonnet 4.6",
    "claude-opus-4-7": "Claude Opus 4.7",
}

MODEL_PALETTE = {
    "gemini-3.5-flash": "#E69F00",
    "gemini-3.1-flash-lite": "#DD8452",
    "gemini-3.1-pro-preview-high-thinking-default": "#B5651D",
    "qwen-3.6-35b-a3b": "#4C72B0",
    "gemma-4": "#CC79A7",
    "openai/gpt-4o": "#4FB04C",
    "claude-sonnet-4-6": "#8172B3",
    "claude-opus-4-7": "#56A0C5",
}
