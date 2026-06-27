import streamlit as st
import asyncio
import sys
import json
import copy
import os
from pathlib import Path
from datetime import datetime

# Add parent directory to path to import patient_simulator
sys.path.insert(0, str(Path(__file__).parent.parent))
from frontend.helpers import initialize_patient
from patient_simulator.misc.llm import LLM, VLLMModelConflictError


st.set_page_config(
    page_title="Virtual Patient Simulator", page_icon="🏥", layout="wide"
)

PROJECT_ROOT = Path(__file__).parent.parent
CHAT_LOG_DIR = PROJECT_ROOT / "results" / "chats"

st.title("🏥 Virtual Patient Simulator")

if "messages" not in st.session_state:
    st.session_state.messages = []

if "prev_settings" not in st.session_state:
    st.session_state.prev_settings = {}

if "chat_start_time" not in st.session_state:
    st.session_state.chat_start_time = None

if "chat_log_path" not in st.session_state:
    st.session_state.chat_log_path = None

if "behavior_reports" not in st.session_state:
    st.session_state.behavior_reports = []

with st.sidebar:
    st.header("Settings")

    conversation_started = st.session_state.get("conversation_started", False)

    st.subheader("Select patient proflie")

    profiles_dir = Path(__file__).parent.parent / "data" / "extracted_profiles_red"
    profile_files = sorted(
        [p.name for p in profiles_dir.glob("*.json") if p.name != "short_desc.json"]
    )
    profile_options = profile_files + ["Custom"]

    selected_profile_name = st.selectbox(
        "Load from extracted profiles",
        profile_options,
        index=0,
        help="Search and load a structured profile from data/extracted_profiles_red",
        disabled=conversation_started,
    )

    if "original_profile_str" not in st.session_state:
        st.session_state.original_profile_str = None
        st.session_state.original_profile_name = None
        st.session_state.display_profile_str = None

    if selected_profile_name != st.session_state.original_profile_name:
        if selected_profile_name != "Custom":
            profile_path = profiles_dir / selected_profile_name
            try:
                raw_text = profile_path.read_text()
                st.session_state.original_profile_str = raw_text
                st.session_state.display_profile_str = json.dumps(
                    json.loads(raw_text), indent=2
                )
                st.session_state.original_profile_name = selected_profile_name
            except Exception as exc:  # noqa: BLE001
                st.error(f"Failed to load profile {selected_profile_name}: {exc}")
                st.session_state.original_profile_str = None
                st.session_state.display_profile_str = None
        else:
            st.session_state.original_profile_str = None
            st.session_state.original_profile_name = "Custom"
            st.session_state.display_profile_str = None

    short_desc_map = {}
    short_desc_path = profiles_dir / "short_desc.json"
    try:
        short_desc_map = json.loads(short_desc_path.read_text())
    except Exception as exc:  # noqa: BLE001
        st.warning(f"Failed to load short descriptions: {exc}")

    selected_profile = None
    selected_profile_str = None
    case_description = None

    if selected_profile_name != "Custom" and st.session_state.display_profile_str:
        try:
            selected_profile = json.loads(st.session_state.original_profile_str)
            selected_profile_str = st.session_state.original_profile_str

            profile_desc = short_desc_map.get(
                selected_profile_name, "No short description available."
            )
            st.markdown(f"**Profile Description:** {profile_desc}")

            with st.expander("Raw Profile JSON"):
                st.text_area(
                    "Profile JSON",
                    value=st.session_state.display_profile_str,
                    height=500,
                    disabled=True,
                    key=f"profile_display_{selected_profile_name}",
                )
        except Exception as exc:  # noqa: BLE001
            st.error(f"Failed to parse stored profile: {exc}")
    elif selected_profile_name == "Custom":
        case_description = st.text_area(
            "Enter patient case vignette in JSON format.",
            value="{}",
            height=300,
            disabled=conversation_started,
        )

    impl_type = st.selectbox(
        "Implementation Type",
        [
            "everyday",
            "craftmd",
            "agentclinic",
            "stateaware",
            "patientsim",
            "virtual",
        ],
        index=0,
        disabled=conversation_started,
    )

    # Show bias selection only for AgentClinic
    bias_present = None
    if impl_type == "agentclinic":
        bias_options = [
            "None",
            "recency",
            "frequency",
            "false_consensus",
            "self_diagnosis",
            "gender",
            "race",
            "sexual_orientation",
            "cultural",
            "education",
            "religion",
            "socioeconomic",
        ]
        bias_present = st.selectbox(
            "Patient Bias",
            bias_options,
            index=0,
            help="Select a cognitive or social bias for the patient",
            disabled=conversation_started,
        )
        if bias_present == "None":
            bias_present = None

    params = None
    meta_llm_config = None

    if impl_type == "patientsim":
        params = {
            "cefr_type": st.selectbox(
                "CEFR Type", ["A", "B", "C"], index=1, disabled=conversation_started
            ),
            "personality_type": st.selectbox(
                "Personality Type",
                [
                    "plain",
                    "verbose",
                    "pleasing",
                    "impatient",
                    "distrust",
                    "overanxious",
                ],
                index=0,
                disabled=conversation_started,
            ),
            "recall_level_type": st.selectbox(
                "Recall Level", ["low", "high"], index=1, disabled=conversation_started
            ),
            "dazed_level_type": st.selectbox(
                "Dazed Level",
                ["normal", "moderate", "high"],
                index=0,
                disabled=conversation_started,
            ),
        }

    elif impl_type == "everyday":
        st.subheader("HEXACO Personality Parameters")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Information Sharing**")
            h = st.select_slider(
                "H - Honesty",
                options=[1, 2, 3],
                value=2,
                format_func=lambda x: [
                    "Honest",
                    "Minimizing",
                    "Deceptive",
                ][x - 1],
                help="Truthfulness about habits and symptoms",
                disabled=conversation_started,
            )
            c = st.select_slider(
                "C - Conscientiousness",
                options=[1, 2, 3],
                value=2,
                format_func=lambda x: [
                    "Precise",
                    "Fuzzy",
                    "Confused",
                ][x - 1],
                help="Memory precision for medical history",
                disabled=conversation_started,
            )

            st.markdown("**Emotional & Social**")
            e = st.select_slider(
                "E - Emotionality",
                options=[1, 2, 3],
                value=2,
                format_func=lambda x: [
                    "Stoic",
                    "Apprehensive",
                    "Distressed",
                ][x - 1],
                help="Emotional demeanor during consultation",
                disabled=conversation_started,
            )
            a = st.select_slider(
                "A - Agreeableness",
                options=[1, 2, 3],
                value=2,
                format_func=lambda x: [
                    "Cooperative",
                    "Neutral",
                    "Hostile",
                ][x - 1],
                help="Cooperation with physician",
                disabled=conversation_started,
            )

        with col2:
            st.markdown("**Cognitive Style**")
            o = st.select_slider(
                "O - Openness",
                options=[1, 2, 3],
                value=2,
                format_func=lambda x: ["Open-Minded", "Skeptical", "Dogmatic"][x - 1],
                help="Openness to medical explanations",
                disabled=conversation_started,
            )
            x = st.select_slider(
                "X - Extraversion",
                options=[1, 2, 3],
                value=2,
                format_func=lambda x: ["Reserved", "Standard", "Talkative"][x - 1],
                help="How much the patient talks",
                disabled=conversation_started,
            )

            st.markdown("**Language**")
            level = st.selectbox(
                "CEFR Level",
                ["A", "B", "C"],
                index=1,
                format_func=lambda x: f"{x} - {['Basic', 'Intermediate', 'Advanced'][['A', 'B', 'C'].index(x)]}",
                help="Language complexity (A=basic, B=intermediate, C=advanced)",
                disabled=conversation_started,
            )

            # dynamic_case_description = st.checkbox(
            #     "Enable Dynamic Case Description",
            #     value=True,
            #     disabled=conversation_started,
            # )

            debug_verbosity = 1

        params = {
            "h": h,
            "e": e,
            "x": x,
            "a": a,
            "c": c,
            "o": o,
            "level": level,
            # "dynamic_case_description": dynamic_case_description,
            "verbosity": debug_verbosity,
        }

    st.subheader("Technical Setup")
    llm_backend = st.selectbox(
        "LLM Backend",
        ["APILLM", "VLLM", "OPENROUTER"],
        index=0,
        disabled=conversation_started,
        help="Choose between APILLM (Gemini), OpenRouter, or local VLLM inference",
    )

    if llm_backend == "APILLM":
        model = st.text_input(
            "Model",
            value="gemini-3.1-flash-lite-preview",
            disabled=conversation_started,
            help="Gemini model identifier",
        )
        with st.expander("APILLM Configuration"):
            st.caption(
                "Sampling controls are only required for VLLM hosting and are configured in the VLLM section."
            )

        llm_config = {
            "vertexai": True,
            "api_key": os.getenv("GOOGLE_CLOUD_API_KEY"),
            "generation_config": {
                "safety_settings": [
                    {
                        "category": "HARM_CATEGORY_HATE_SPEECH",
                        "threshold": "OFF",
                    },
                    {
                        "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                        "threshold": "OFF",
                    },
                    {
                        "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                        "threshold": "OFF",
                    },
                    {
                        "category": "HARM_CATEGORY_HARASSMENT",
                        "threshold": "OFF",
                    },
                ],
                "thinking_config": {
                    "thinking_level": "LOW",
                },
            },
        }
    elif llm_backend == "OPENROUTER":
        model = st.text_input(
            "Model",
            value="deepseek/deepseek-v3.2",
            disabled=conversation_started,
            help="OpenRouter model identifier",
        )
        llm_config = None
    else:
        model = st.text_input(
            "Model Path/HF ID",
            value="Qwen/Qwen3-4B-Instruct-2507",
            disabled=conversation_started,
            help="HuggingFace model ID or local path",
        )

        with st.expander("VLLM Configuration"):
            tensor_parallel_size = st.number_input(
                "Tensor Parallel Size",
                min_value=1,
                max_value=8,
                value=1,
                disabled=conversation_started,
            )
            gpu_memory_utilization = st.slider(
                "GPU Memory Utilization",
                min_value=0.1,
                max_value=0.95,
                value=0.85,
                step=0.05,
                disabled=conversation_started,
            )
            max_model_len = st.number_input(
                "Max Model Length",
                min_value=512,
                max_value=16384,
                value=4000,
                disabled=conversation_started,
            )
            temperature = st.slider(
                "Temperature",
                min_value=0.0,
                max_value=2.0,
                value=0.5,
                step=0.1,
                disabled=conversation_started,
            )
            top_p = st.slider(
                "Top P",
                min_value=0.0,
                max_value=1.0,
                value=0.95,
                step=0.01,
                disabled=conversation_started,
            )
            repetition_penalty = st.slider(
                "Repetition Penalty",
                min_value=1.0,
                max_value=2.0,
                value=1.2,
                step=0.1,
                disabled=conversation_started,
            )

            max_output_tokens = st.number_input(
                "Max Output Tokens",
                min_value=128,
                max_value=4096,
                value=256,
                disabled=conversation_started,
            )

            st.divider()
            if st.button(
                "🔄 Kill VLLM Process",
                help="Kill the current VLLM process to free GPU memory",
            ):
                import subprocess

                try:
                    subprocess.run(["pkill", "-f", "VLLM::EngineCore"], check=False)
                    st.success("VLLM process killed successfully")
                    if "patient" in st.session_state and hasattr(
                        st.session_state.patient, "llm"
                    ):
                        if hasattr(st.session_state.patient.llm, "cleanup"):
                            st.session_state.patient.llm.cleanup()
                    st.session_state.patient = None
                except Exception as e:
                    st.error(f"Error killing VLLM process: {e}")

        llm_config = {
            "engine_kwargs": {
                "dtype": "auto",
                "tensor_parallel_size": tensor_parallel_size,
                "gpu_memory_utilization": gpu_memory_utilization,
                "max_model_len": max_model_len,
            },
            "sampling_kwargs": {
                "temperature": temperature,
                "top_p": top_p,
                "repetition_penalty": repetition_penalty,
                "max_tokens": max_output_tokens,
            },
        }

    meta_llm_config = None
    meta_llm_backend = st.selectbox(
        "Meta-LLM Backend",
        ["APILLM", "VLLM", "OPENROUTER"],
        index=1,
        disabled=conversation_started,
        key="meta_llm_backend",
        help="Choose between APILLM (Gemini), OpenRouter, or local VLLM inference",
    )

    if meta_llm_backend == "APILLM":
        meta_model = st.text_input(
            "Meta Model",
            value="gemini-3.1-flash-lite-preview",
            disabled=conversation_started,
            key="meta_model",
            help="Gemini model identifier for meta-cognitive tasks",
        )
        with st.expander("Meta-APILLM Configuration"):
            st.caption(
                "Sampling controls are only required for VLLM hosting and are configured in the Meta-VLLM section."
            )

        meta_runtime_config = {
            "vertexai": True,
            "api_key": os.getenv("GOOGLE_CLOUD_API_KEY"),
            "generation_config": {
                "safety_settings": [
                    {
                        "category": "HARM_CATEGORY_HATE_SPEECH",
                        "threshold": "OFF",
                    },
                    {
                        "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                        "threshold": "OFF",
                    },
                    {
                        "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                        "threshold": "OFF",
                    },
                    {
                        "category": "HARM_CATEGORY_HARASSMENT",
                        "threshold": "OFF",
                    },
                ],
                "thinking_config": {
                    "thinking_level": "LOW",
                },
            },
        }
    elif meta_llm_backend == "OPENROUTER":
        meta_model = st.text_input(
            "Meta Model",
            value="deepseek/deepseek-v3.2",
            disabled=conversation_started,
            key="meta_model",
            help="OpenRouter model identifier for meta-cognitive tasks",
        )
        meta_runtime_config = None
    else:
        meta_model = st.text_input(
            "Meta Model Path/HF ID",
            value="mistralai/Ministral-3-14B-Instruct-2512",
            disabled=conversation_started,
            key="meta_model",
        )
        with st.expander("Meta-VLLM Configuration"):
            meta_temperature = st.slider(
                "Temperature",
                min_value=0.0,
                max_value=2.0,
                value=0.05,
                step=0.05,
                disabled=conversation_started,
                key="meta_temperature",
            )
            meta_top_p = st.slider(
                "Top P",
                min_value=0.0,
                max_value=1.0,
                value=0.95,
                step=0.01,
                disabled=conversation_started,
                key="meta_top_p",
            )
            meta_max_output_tokens = st.number_input(
                "Max Output Tokens",
                min_value=128,
                max_value=4096,
                value=512,
                disabled=conversation_started,
                key="meta_max_output_tokens",
            )
        meta_runtime_config = {
            "batch_size": 16,
            "engine_kwargs": {
                "dtype": "auto",
                "tensor_parallel_size": 1,
                "gpu_memory_utilization": 0.85,
                "max_model_len": 8192,
            },
            "sampling_kwargs": {
                "temperature": meta_temperature,
                "top_p": meta_top_p,
                "max_tokens": meta_max_output_tokens,
            },
        }

    meta_llm_config = {
        "model": meta_model,
        "backend": meta_llm_backend,
        "runtime_config": meta_runtime_config,
    }

    if st.button("Reset Conversation", type="primary", width="stretch"):
        st.session_state.messages = []
        st.session_state.behavior_reports = []
        st.session_state.chat_start_time = None
        st.session_state.chat_log_path = None
        st.session_state.patient = None
        st.session_state.prev_settings = {}
        st.session_state.conversation_started = False
        LLM._response_cache.clear()
        st.rerun()

# Create a deep copy to prevent the patient from modifying the displayed profile
patient_payload = (
    copy.deepcopy(selected_profile)
    if selected_profile is not None
    else case_description
)

case_description_value = (
    selected_profile_str if selected_profile_str is not None else case_description or ""
)

current_settings = {
    "case_description": case_description_value,
    "impl_type": impl_type,
    "model": model,
    "llm_backend": llm_backend,
    "llm_config": str(llm_config) if llm_config else None,
    "bias_present": bias_present if impl_type == "agentclinic" else None,
    "selected_profile": selected_profile_name,
    "params": tuple(sorted(params.items())) if params else None,
    "meta_llm_config": str(meta_llm_config) if meta_llm_config else None,
}


def _sanitize_for_logging(value):
    if isinstance(value, dict):
        sanitized = {}
        for key, item in value.items():
            key_lower = str(key).lower()
            if "key" in key_lower or "token" in key_lower or "secret" in key_lower:
                sanitized[key] = "REDACTED"
            else:
                sanitized[key] = _sanitize_for_logging(item)
        return sanitized
    if isinstance(value, list):
        return [_sanitize_for_logging(item) for item in value]
    return value


logging_settings = {
    "impl_type": impl_type,
    "model": model,
    "llm_backend": llm_backend,
    "llm_config": _sanitize_for_logging(llm_config),
    "meta_llm_config": _sanitize_for_logging(meta_llm_config),
    "params": _sanitize_for_logging(params),
    "bias_present": bias_present if impl_type == "agentclinic" else None,
    "selected_profile": selected_profile_name,
    "case_description": case_description_value,
}


def _ensure_chat_log_path() -> Path:
    if st.session_state.chat_log_path is None:
        start_time = datetime.now()
        st.session_state.chat_start_time = start_time.isoformat(timespec="seconds")
        CHAT_LOG_DIR.mkdir(parents=True, exist_ok=True)
        file_name = f"{start_time.strftime('%Y%m%d_%H%M%S_%f')}.json"
        st.session_state.chat_log_path = str(CHAT_LOG_DIR / file_name)
    return Path(st.session_state.chat_log_path)


def _write_chat_log() -> None:
    log_path = _ensure_chat_log_path()
    payload = {
        "chat_start_time": st.session_state.chat_start_time,
        "last_updated": datetime.now().isoformat(timespec="seconds"),
        "settings": logging_settings,
        "turns": st.session_state.messages,
        "reports": st.session_state.behavior_reports,
    }
    log_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2))


def _init_patient():
    try:
        with st.spinner("initializing patient..."):
            st.session_state.patient = initialize_patient(
                impl_type,
                patient_payload,
                model,
                bias_present,
                params,
                llm_backend,
                llm_config,
                meta_llm_config=meta_llm_config,
            )
    except VLLMModelConflictError as e:
        st.session_state.patient = None
        st.error(str(e))
    except RuntimeError as e:
        error_text = str(e)
        if "vllm failed to start after recovery attempt" in error_text.lower():
            st.session_state.patient = None
            st.error(error_text)
            return
        raise


if st.session_state.prev_settings != current_settings:
    _init_patient()
    st.session_state.messages = []
    st.session_state.behavior_reports = []
    st.session_state.chat_start_time = None
    st.session_state.chat_log_path = None
    st.session_state.prev_settings = current_settings

if "patient" not in st.session_state or st.session_state.patient is None:
    _init_patient()

if st.session_state.get("patient") is None:
    st.warning(
        "Provide a valid case description or select an extracted profile to start."
    )
    st.stop()

if st.session_state.get("patient") is None:
    st.warning(
        "Provide a valid case description or select an extracted profile to start."
    )
    st.stop()

if impl_type == "everyday":
    patient = st.session_state.patient

    if patient.prior_belief:
        st.info(f"💭 Prior Belief: {patient.prior_belief}")
    if patient.current_emotional_state:
        st.info(f"😟 Emotional State: {patient.current_emotional_state}")
    if patient.tangent_topic:
        st.info(f"💬 Tangent Topic: {patient.tangent_topic}")

# State descriptions for state-aware patient
STATE_DESCRIPTIONS = {
    "initialization": "🔵 Initialization",
    "A-A-A": "🟢 Effective Inquiry (with info)",
    "A-A-B": "🟡 Ineffective Inquiry (no info)",
    "A-B": "🟠 Ambiguous Inquiry (too broad)",
    "B-A-A": "🟢 Effective Advice (with results)",
    "B-A-B": "🟡 Ineffective Advice (no results)",
    "B-B": "🟠 Ambiguous Advice (too broad)",
    "C": "🔴 Demand (physical action)",
    "D": "🔴 Other Topics (off-topic)",
    "E": "⚫ Conclusion (end)",
}


def display_state_metadata(message):
    """Display state code and instruction for state-aware patient messages."""
    if message.get("state_code"):
        st.caption(
            f"State: {STATE_DESCRIPTIONS.get(message['state_code'], message['state_code'])}"
        )
    if message.get("instruction"):
        st.caption(f"Instruction: {message['instruction']}")


def display_everyday_metadata(message, patient):
    """Display metadata for everyday patient messages."""
    if message.get("relevant_fields"):
        st.caption(f"📋 Affected Fields: {', '.join(message['relevant_fields'])}")


def get_current_system_prompt(patient, impl_type):
    """Get the current system prompt from the patient."""
    if impl_type in ["stateaware"]:
        return None
    elif impl_type == "everyday":
        return patient._build_system_prompt()
    else:
        return getattr(patient, "system_instruction", None)


def get_latest_assistant_system_prompt(messages):
    """Get the most recent assistant system prompt from chat history."""
    for message in reversed(messages):
        if message["role"] == "assistant" and message.get("system_prompt"):
            return message["system_prompt"]
    return None


def display_prompt_changes(old_prompt, new_prompt):
    """Display what changed between two prompts with word-level highlighting."""
    if old_prompt == new_prompt or not old_prompt:
        return

    import difflib

    old_words = old_prompt.split()
    new_words = new_prompt.split()

    matcher = difflib.SequenceMatcher(None, old_words, new_words)

    added_words = []
    removed_words = []

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "replace":
            removed_words.extend(old_words[i1:i2])
            added_words.extend(new_words[j1:j2])
        elif tag == "delete":
            removed_words.extend(old_words[i1:i2])
        elif tag == "insert":
            added_words.extend(new_words[j1:j2])


def _send_user_message(prompt: str) -> None:
    prompt = prompt.strip()
    if not prompt:
        st.warning("Please enter a message before sending.")
        return

    st.session_state.conversation_started = True
    st.session_state.messages.append({"role": "user", "content": prompt})

    patient = st.session_state.patient
    with st.spinner("generating patient answer..."):
        response = asyncio.run(patient.get_response(prompt))

    message_data = {"role": "assistant", "content": response}

    current_system_prompt = get_current_system_prompt(patient, impl_type)
    if current_system_prompt:
        message_data["system_prompt"] = current_system_prompt

    if impl_type == "stateaware" and hasattr(patient, "last_state_code"):
        message_data["state_code"] = patient.last_state_code
        message_data["instruction"] = patient.current_instruction
    elif impl_type == "everyday":
        if patient.current_emotional_state:
            message_data["emotional_state"] = patient.current_emotional_state

        if patient.tangent_topic:
            message_data["tangent_topic"] = patient.tangent_topic

        if hasattr(patient, "last_relevant_fields") and patient.last_relevant_fields:
            message_data["relevant_fields"] = patient.last_relevant_fields

    st.session_state.messages.append(message_data)
    _write_chat_log()
    st.rerun()


def _submit_report(report_text: str) -> None:
    report_text = report_text.strip()
    if not report_text:
        st.warning("Please enter a report message before submitting.")
        return

    st.session_state.conversation_started = True
    report_entry = {
        "time": datetime.now().isoformat(timespec="seconds"),
        "text": report_text,
        "turn_index": len(st.session_state.messages),
    }
    st.session_state.behavior_reports.append(report_entry)
    st.session_state.messages.append(
        {
            "role": "user",
            "content": f"[REPORT] {report_text}",
            "is_report": True,
        }
    )
    _write_chat_log()
    st.rerun()


@st.dialog("Report issue with current chat")
def report_issue_dialog() -> None:
    report_text = st.text_area(
        "Describe the issue",
        placeholder="Describe what went wrong in this chat...",
        key="report_issue_text",
    )
    submit_col, cancel_col = st.columns(2)
    with submit_col:
        if st.button("Submit Report", type="primary", width="stretch"):
            _submit_report(report_text)
    with cancel_col:
        if st.button("Cancel", width="stretch"):
            st.rerun()


show_everyday_prompt_pane = impl_type == "everyday"
chat_container = st.container()
prompt_pane_placeholder = None

if show_everyday_prompt_pane:
    chat_col, prompt_col = st.columns([3, 2], gap="large")
    chat_container = chat_col
    with prompt_col:
        st.subheader("System Prompt")
        prompt_pane_placeholder = st.empty()

if st.session_state.messages or st.session_state.get("patient") is not None:
    with chat_container:
        if impl_type != "everyday":
            st.subheader("System Prompt")
            patient = st.session_state.patient
            current_prompt = get_current_system_prompt(patient, impl_type)

            if current_prompt:
                st.text(current_prompt)

        conversation_col, report_col = st.columns([5, 2])
        with conversation_col:
            st.subheader("💬 Conversation")
        with report_col:
            if st.button("🚩 Report Issue within chat", width="stretch"):
                report_issue_dialog()

if show_everyday_prompt_pane and prompt_pane_placeholder is not None:
    latest_prompt = get_latest_assistant_system_prompt(st.session_state.messages)
    if latest_prompt:
        prompt_pane_placeholder.text(latest_prompt)
    else:
        prompt_pane_placeholder.write(
            "Ask the patient a question to see the system prompt used to generate the answer"
        )

with chat_container:
    for i, message in enumerate(st.session_state.messages):
        with st.chat_message(message["role"]):
            if message["role"] == "assistant":
                if message.get("system_prompt"):
                    prev_prompt = None
                    for j in range(i - 1, -1, -1):
                        if st.session_state.messages[j][
                            "role"
                        ] == "assistant" and st.session_state.messages[j].get(
                            "system_prompt"
                        ):
                            prev_prompt = st.session_state.messages[j]["system_prompt"]
                            break
                    display_prompt_changes(prev_prompt, message["system_prompt"])

                if impl_type == "stateaware":
                    display_state_metadata(message)
                elif impl_type == "everyday":
                    display_everyday_metadata(message, st.session_state.patient)
            st.markdown(message["content"])

    prompt = st.chat_input("Ask the patient a question...")
    if prompt is not None:
        _send_user_message(prompt)

st.caption("Note, that your chats are logged for research purposes.")
