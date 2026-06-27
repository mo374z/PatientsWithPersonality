# Patient Simulator Comparison and Evaluation

We aim to implement and compare different patient simulator approaches proposed in current papers in order to compare them with each other and evaluate their capabilities.

## Patient Simulator Implementations

### 1. BaselinePatient
Rephrases the real patient responses from the transcript using an LLM. Serves as a lower-bound baseline — it has access to the ground-truth answers.

### 2. VirtualPatient (EasyMED)
Our primary implementation. Single-turn system prompt rebuilt each turn with a JSON-formatted case and a rolling conversation history window. Responds at temperature 0.3.

### 3. CraftMDPatient (Johri et al., 2023)
Simple prompt-based simulation with conversation history. Responses are kept to 1 sentence in layman's terminology.

### 4. AgentClinicPatient (Schmidgall et al., 2024)
Dialogue-based simulation supporting 11 types of cognitive and social biases (gender, race, self-diagnosis, etc.).

### 5. StateAwarePatient (Liao et al., 2024)
State-aware simulation with a three-tier memory system and precise behavioral control.

**Architecture:**
- **State Tracker**: Classifies doctor actions into 8 states (A-A-A, A-A-B, A-B, B-A-A, B-A-B, B-B, C, D)
  - A: Inquiry (Specific with info / Specific without info / Ambiguous)
  - B: Advice (Specific with results / Specific without results / Ambiguous)
  - C: Demand (physical actions)
  - D: Other Topics (off-topic)
- **Memory Bank**: Long-term (patient info), working (response requirements), short-term (dialogue history)
- **Response Generator**: Context-aware responses grounded in patient data to prevent hallucination

### 6. PatientSimPatient (Kyung et al., 2025)
Richly parameterized simulation across four independent axes:
- **CEFR language level**: A / B / C (with matched medical vocabulary lists)
- **Personality**: plain, verbose, pleasing, impatient, distrust, overanxious
- **Memory recall**: low / high
- **Dazedness**: normal / moderate / high (fades progressively over the conversation)

### 7. PatientsWithPersonality (HEXACO-based, ours)
Realistic information recall and disclosure simulator grounded in the HEXACO personality model.

**Architecture:**
- **HEXACO traits** (each scored 1–3): Honesty-Humility, Emotionality, Extraversion, Agreeableness, Conscientiousness, Openness
- **Dynamic disclosure**: leisure fields (tobacco, alcohol, drugs, etc.) drawn from a 3-column grid — truthful / downplayed / denied — based on Honesty-Humility score
- **Fuzzy medical history**: Conscientiousness score controls how precisely the patient recalls their medical history (exact → approximate → vague)
- **Lazy initialization**: prior beliefs (Openness), emotional state (Emotionality), tangent topics (Extraversion), and a latent role description are generated once on first call via a meta-LLM
- **Per-turn field classification**: a meta-LLM classifies which case fields are relevant to each doctor question, allowing the disclosure level to evolve across the conversation

## GUIs

### Labeling Frontend

```bash
uv run python labeling-gui/app.py \
  --results-dir results/patient_comparison_default/ \
  --study-config labeling-gui/study_config.yaml \
  --data-dir data/ \
```

Both apps run independently. Adjust the patient names and case lists in the respective `study_config_*.yaml` files before starting. Labels are stored under `labeling-gui/labels_realism/` and `labeling-gui/labels_personality/` by default (override with `--labels-dir`).

To run the streamlit GUI use the commands:

```bash
uv run streamlit run frontend/evaluation_explorer.py
```
