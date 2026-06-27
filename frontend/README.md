# Running the Streamlit Frontend

## Start the main simulator app

```bash
uv run streamlit run frontend/app.py --server.port 8555
```

You can also open this to the public via a tailscale funnel

```bash
tailscale funnel --bg --https 8443 8555
```

Turn the funnel off by using

```bash
tailscale funnel --https=8443 off
```

## Start the evaluation explorer

```bash
uv run streamlit run frontend/evaluation_explorer.py
```

## Main App Features

- **LLM Backend Selection**: Choose between Gemini API or local VLLM inference
- **Model Selection**:
  - For Gemini API: Choose between gemini-2.5-flash and gemini-2.5-pro
  - For VLLM: Enter HuggingFace model ID or local path (e.g., `Qwen/Qwen3-4B-Instruct-2507`)
- **Implementation Type**: Toggle between different patient implementation types
  - **CraftMD**: Simple 1-sentence responses
  - **AgentClinic**: Detailed dialogue with cognitive/social biases
  - **StateAware**: State-tracking with memory bank
  - **PatientSim**: Persona-controlled patient simulation
  - **Everyday**: Realistic patient with HEXACO personality framework
- **Everyday Patient HEXACO Configuration**: When using Everyday implementation, configure:
  - **H - Honesty**: Truthfulness about habits (Honest → Deceptive)
  - **E - Emotionality**: Emotional demeanor (Stoic → Volatile)
  - **X - Extraversion**: How much patient talks (Talkative → Reserved)
  - **A - Agreeableness**: Cooperation level (Cooperative → Hostile)
  - **C - Conscientiousness**: Memory precision (Precise → Confused)
  - **O - Openness**: Flexibility of thinking (Open-Minded → Dogmatic)
  - **CEFR Level**: Language complexity (A1-C2)
  - **Debug Verbosity**: Console output detail (0-2)
- **VLLM Configuration**: Fine-tune local inference parameters
  - Tensor parallel size, GPU memory utilization
  - Max model length, temperature, max tokens
- **Case Description**: Load from extracted profiles or enter custom case vignette
- **Chat Interface**: Natural conversation with the virtual patient
- **Metadata Display**: See state codes, emotional states, and relevant fields for each response
- **Reset Button**: Clear conversation and start fresh

## Evaluation Explorer Features

- **Patient Type Selection**: Browse different patient implementations (CraftMD, AgentClinic, StateAware, PatientSim, etc.)
- **Conversation Selection**: Choose from available evaluated conversations
- **Turn-by-Turn View**: See doctor questions with both real and simulated patient responses side-by-side
- **Detailed Metrics**: View relevance, realism (content & style), sentiment, and token ratio metrics for each turn
- **Overall Statistics**: Sidebar displays conversation-level metrics including persona consistency, domain term usage, and aggregate scores

## Usage

### Main App

1. Configure your settings in the sidebar
2. Enter or modify the patient case description
3. Start asking questions in the chat interface
4. The patient will respond based on their case vignette

### Evaluation Explorer

1. Select a patient type from the sidebar dropdown
2. Choose a conversation to explore
3. Review each turn's doctor question, real response, and simulated response
4. Expand turn metrics to see detailed evaluation scores
5. Check the sidebar for overall conversation statistics
