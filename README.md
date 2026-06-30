<div align="center">

# 🩺 Patients With Personality

### Realistic Patient Simulation through Controlled Diversity and Selective Disclosure

[Moritz Schlager](https://scholar.google.com/citations?user=lMvbxhoAAAAJ&hl=en), [Friederike Jungmann](https://kiinformatik.mri.tum.de/de/team/friederike_jungmann), [Samuel Schmidgall](https://scholar.google.com/citations?user=bQDooZEAAAAJ&hl=en), Philipp Raffler, Franziska Hartl, Eva Wende, Paula Roßmüller, Conrad Ketzer, [Avinatan Hassidim](https://scholar.google.com/citations?user=CnBvgwcAAAAJ&hl=en), [Dale R. Webster](https://scholar.google.com/citations?user=qAqGfk0AAAAJ&hl=en), [Yossi Matias](https://scholar.google.com/citations?user=IwSe1-MAAAAJ&hl=en), [Yun Liu](https://scholar.google.com/citations?user=EojZy50AAAAJ&hl=en), [Daniel Rueckert](https://scholar.google.com/citations?user=H0O0WnQAAAAJ&hl=en), [Mike Schaekermann](https://scholar.google.com/citations?user=mwj_ldQAAAAJ&hl=en), [Paul Hager](https://scholar.google.com/citations?user=ESLUtGAAAAAJ&hl=en)

<p>

[![Paper](https://img.shields.io/badge/arXiv-2606.17441-b31b1b.svg?style=flat)](https://arxiv.org/abs/2606.17441)
[![Project Page](https://img.shields.io/badge/Project-Page-1f9e87.svg?style=flat)](https://mo374z.github.io/PatientsWithPersonality/)
[![Python 3.12](https://img.shields.io/badge/Python-3.12-3776AB.svg?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![uv](https://img.shields.io/badge/managed%20by-uv-DE5FE9.svg?style=flat)](https://github.com/astral-sh/uv)

</p>

</div>

---

This repository contains the full code for the paper: the **PWP simulator**, **re-implementations of five prior patient simulators** under a shared interface, the **Hydra-configured evaluation and benchmarking pipeline**, and the **Streamlit/Dash GUIs** used in the clinician study.

## Motivation

<p align="center">
  <img src="docs/static/images/motivation.png" width="80%" alt="Oversharing baselines vs. PWP selective disclosure">
</p>

Realistic patient simulation is needed to benchmark clinical LLMs at scale without time-consuming, expensive user studies. Yet existing simulators behave unlike real patients: prompted with a single question, they dump the entire case — diagnoses, medications, family history, lifestyle — in one overly cooperative turn. Real patients disclose selectively, recall imperfectly, and vary widely in how they communicate. PWP closes this gap by parametrizing patient behavior over personality, so a virtual patient reveals only what is asked, the way a real person would.

## Highlights

- 🧬 **HEXACO-grounded personas** — six personality axes (Honesty-Humility, Emotionality, Extraversion, Agreeableness, Conscientiousness, Openness) give fine-grained, recoverable control over how a patient behaves.
- 🤐 **Selective disclosure** — patients reveal information only when prompted, preventing the unprompted oversharing that plagues prior simulators.
- 🎭 **Controlled diversity** — configured traits span a substantially wider behavioral footprint than the closest baseline.
- 👩‍⚕️ **Clinician-validated** — judged nearly as realistic as recorded human actors in a blinded clinician study.
- 🔬 **Reproducible benchmark** — Hydra-configured experiment suite spanning multiple simulators and LLM backends.

## Framework

<p align="center">
  <img src="docs/static/images/framework.png" width="100%" alt="PWP framework overview">
</p>

PWP separates a one-time **initialization** from per-turn **response generation**, mediated by a **latent patient state**.

**Initialization.** A case description (personal background, lifestyle habits, medical facts) and a personality parametrization (HEXACO traits + CEFR language level) are combined by meta-LLM operations into:

- a **behavioral role** — tangent topics drawn from personal information, an emotional state driven by Emotionality (E), and prior beliefs driven by Openness (O);
- a **disclosure grid** — lifestyle fields in truthful / downplayed / denied variants gated by Honesty-Humility (H), and medical facts in original / fuzzy / omitted variants gated by Conscientiousness (C).

**Response generation.** For each clinician question, a meta-LLM classifies which case field is being requested, the latent state selects the appropriate disclosure variant, and the conversational LLM generates the final answer. This repeats across the multi-turn conversation, so disclosure evolves naturally and the patient never overshares.

## Results

<p align="center">
  <img src="docs/static/images/realism_subscores.png" width="100%" alt="Clinician realism evaluation">
</p>

In a blinded clinician study, PWP is judged nearly as realistic as recorded human actors and clearly ahead of prior simulators, while being flagged as "too informative" far less often. Full analyses — selective disclosure, HEXACO steerability, and conversational diversity — are on the [project page](https://mo374z.github.io/PatientsWithPersonality/).

## Patient Simulators

The repository implements PWP alongside re-implementations of prior simulators under a common interface in [`patient_simulator/patients/`](patient_simulator/patients/).

| Simulator | Reference | Summary |
|---|---|---|
| **PatientsWithPersonality** | *Ours* | HEXACO-grounded realistic recall and selective disclosure over a latent patient state. |
| `BaselinePatient` | — | Rephrases the ground-truth transcript responses. Lower-bound baseline with access to gold answers. |
| `VirtualPatient` | EasyMED | Single-turn system prompt rebuilt each turn from a JSON case and a rolling history window. |
| `CraftMDPatient` | Johri et al., 2023 | Prompt-based simulation; responses kept to one layman sentence. |
| `AgentClinicPatient` | Schmidgall et al., 2024 | Dialogue simulation supporting 11 cognitive and social biases. |
| `StateAwarePatient` | Liao et al., 2024 | State tracker + three-tier memory bank for precise behavioral control. |
| `PatientSimPatient` | Kyung et al., 2025 | Persona axes: CEFR level, personality, memory recall, dazedness. |

<details>
<summary><b>Architecture details</b></summary>

### PatientsWithPersonality (ours)
- **HEXACO traits** (each scored 1–3): Honesty-Humility, Emotionality, Extraversion, Agreeableness, Conscientiousness, Openness.
- **Dynamic disclosure**: leisure fields (tobacco, alcohol, drugs, …) drawn from a three-column grid — truthful / downplayed / denied — based on the Honesty-Humility score.
- **Fuzzy medical history**: Conscientiousness controls recall precision (exact → approximate → vague).
- **Lazy initialization**: prior beliefs (Openness), emotional state (Emotionality), tangent topics (Extraversion), and a latent role description are generated once on first call via a meta-LLM.
- **Per-turn field classification**: a meta-LLM classifies which case fields are relevant to each doctor question, letting disclosure evolve across the conversation.

### StateAwarePatient (Liao et al., 2024)
- **State Tracker**: classifies doctor actions into 8 states (A-A-A, A-A-B, A-B, B-A-A, B-A-B, B-B, C, D).
  - A: Inquiry · B: Advice · C: Demand (physical actions) · D: Other Topics.
- **Memory Bank**: long-term (patient info), working (response requirements), short-term (dialogue history).
- **Response Generator**: context-aware responses grounded in patient data to prevent hallucination.

### PatientSimPatient (Kyung et al., 2025)
- **CEFR language level**: A / B / C with matched medical vocabulary lists.
- **Personality**: plain, verbose, pleasing, impatient, distrust, overanxious.
- **Memory recall**: low / high · **Dazedness**: normal / moderate / high (fades over the conversation).

</details>

## Installation

The project uses [`uv`](https://github.com/astral-sh/uv) and targets Python 3.12.

```bash
git clone https://github.com/mo374z/PatientsWithPersonality.git
cd PatientsWithPersonality
uv sync
```

Provide API credentials by copying the example key file and filling in your values:

```bash
cp keys.json.example keys.json
```

## Quickstart

Experiments are configured with [Hydra](https://hydra.cc/); configs live in [`configs/experiment/`](configs/experiment/).

```bash
# Compare simulators on the default case set
uv run python scripts/run_patient_comparison.py

# Run the downstream diagnostic benchmark
uv run python scripts/run_helpmed_benchmark.py
```

| Script | Purpose |
|---|---|
| `scripts/run_patient_comparison.py` | Run and evaluate all simulators side by side. |
| `scripts/run_helpmed_benchmark.py` | Downstream diagnostic benchmark across LLM backends. |
| `scripts/run_metaprompt_tuning.py` | Tune the meta-prompts driving the simulator. |
| `scripts/run_posthoc_eval.py` | Re-score existing conversation results. |
| `scripts/run_llm_feasibility.py` | Probe LLM-backend feasibility. |

Model backends are swappable via the configs under [`configs/experiment/models/`](configs/experiment/models/) (Claude, Gemini, GPT, Qwen, Ministral, …).

## Interactive GUIs

### Simulator playground & evaluation explorer (Streamlit)

```bash
# Chat live with a configurable virtual patient
uv run streamlit run frontend/app.py --server.port 8555

# Inspect evaluated conversations turn by turn
uv run streamlit run frontend/evaluation_explorer.py
```

See [`frontend/README.md`](frontend/README.md) for the full feature list and the HEXACO configuration controls.

### Clinician labeling GUI (Dash)

```bash
uv run python labeling-gui/app.py \
  --results-dir results/patient_comparison_default/ \
  --study-config labeling-gui/study_config.yaml \
  --data-dir data/
```

Adjust patient names and case lists in the `study_config_*.yaml` files before starting. Labels are written to `labeling-gui/labels_realism/` and `labeling-gui/labels_personality/` (override with `--labels-dir`).

## Repository Structure

```
patient_simulator/      Core library
  patients/             Simulator implementations (PWP + baselines)
  prompts/              Patient and evaluation prompts
  misc/                 LLM clients, plotting, metrics, utilities
  eval.py               Evaluation pipeline
configs/experiment/     Hydra experiment + model configs
scripts/                Experiment entry points
frontend/               Streamlit playground & evaluation explorer
labeling-gui/           Dash clinician labeling app
notebooks/              Analysis notebooks and paper figures
tests/                  Tests
```

## Citation

If you find this work useful, please cite:

```bibtex
@article{schlager2026patients,
  title   = {Patients With Personality: Realistic Patient Simulation through Controlled Diversity and Selective Disclosure},
  author  = {Schlager, Moritz and Jungmann, Friederike and Schmidgall, Samuel and Raffler, Philipp and Hartl, Franziska and Wende, Eva and Ro{\ss}m{\"u}ller, Paula and Ketzer, Conrad and Hassidim, Avinatan and Webster, Dale R. and Matias, Yossi and Liu, Yun and Rueckert, Daniel and Schaekermann, Mike and Hager, Paul},
  journal = {arXiv preprint arXiv:2606.17441},
  year    = {2026}
}
```
