from typing import List, Tuple, Dict, Any, Optional
import pydantic
import logging
import pandas as pd
import csv
import json
from tqdm import tqdm
from readability import Readability
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from patient_simulator.misc.llm import LLM
from patient_simulator.data_extraction import extract_patient_profile_from_text
from patient_simulator.prompts.eval_prompts import (
    EVAL_CASE_EXTRACTION_PROMPT,
    PROFILE_FIDELITY_PROMPT,
    FACT_FIDELITY_PROMPT,
    PERSONAL_INFO_PRESENCE_PROMPT,
    RELEVANCE_EVALUATION_PROMPT,
    REALISM_CONTENT_EVALUATION_PROMPT,
    REALISM_STYLE_EVALUATION_PROMPT,
    PERSONA_CONSISTENCY_EVALUATION_PROMPT,
    PERSONALITY_RECON_H_PROMPT,
    PERSONALITY_RECON_E_PROMPT,
    PERSONALITY_RECON_X_PROMPT,
    PERSONALITY_RECON_A_PROMPT,
    PERSONALITY_RECON_C_PROMPT,
    PERSONALITY_RECON_O_PROMPT,
    TAGGED_FACT_EXTRACTION_PROMPT,
    REALISM_JUDGE_PROMPT,
    CaseDescription,
)
from patient_simulator.prompts.patient_prompts import PWP_CLASS
from patient_simulator.patients.pwp import (
    LEISURE_FIELDS,
    MEDICAL_FIELDS,
    StringFields,
)
from patient_simulator.misc.utils import extract_tagged_text

logging.getLogger().setLevel(logging.WARNING)

try:
    import nltk

    nltk.data.find("tokenizers/punkt_tab")
except LookupError:
    logging.info("Downloading NLTK punkt_tab tokenizer...")
    import nltk

    nltk.download("punkt_tab", quiet=True)


class _AtomicFactsExtraction(pydantic.BaseModel):
    """Structured extraction container for patient atomic facts."""

    facts: list[str] = pydantic.Field(
        default_factory=list,
        description="Atomic patient facts as a list of single, checkable statements.",
    )


class _FactFidelityBatch(pydantic.BaseModel):
    """Structured support flags aligned with atomic fact positions."""

    supported: list[bool] = pydantic.Field(
        default_factory=list,
        description="Boolean support labels in the same order as provided facts.",
    )


class _PersonalInfoPresenceBatch(pydantic.BaseModel):
    """Structured presence flags aligned with personal-info field positions."""

    contained: list[bool] = pydantic.Field(
        default_factory=list,
        description="Boolean containment labels in the same order as provided fields.",
    )


class _TaggedFact(pydantic.BaseModel):
    text: str
    field: str
    supported: bool


class _TaggedFactsExtraction(pydantic.BaseModel):
    facts: list[_TaggedFact] = pydantic.Field(default_factory=list)


class _RealismJudgeOutput(pydantic.BaseModel):
    p_real: pydantic.confloat(ge=0.0, le=1.0)
    symptom_realism: pydantic.conint(ge=1, le=5)
    information_control: pydantic.conint(ge=1, le=5)
    style_realism: pydantic.conint(ge=1, le=5)
    justification: str


def _load_domain_terms(
    csv_path: str = "data/CHV_concepts_terms.csv",
) -> Dict[str, float]:
    """Load medical domain terms and their combo scores into a dict for fast lookup."""
    terms_dict = {}
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            term = str(row["Term"]).strip().lower()
            combo_score = float(row["Combo Score"])
            terms_dict[term] = combo_score
    return terms_dict


def _count_domain_terms(
    text: str, terms_dict: Dict[str, float], fpdigits: int = 3
) -> Tuple[int, float]:
    """Count unique medical domain terms in text and compute average combo score."""
    words = text.lower().split()
    found_terms = []
    found_scores = []

    for word in words:
        word_clean = word.strip(".,!?;:()[]\"'")
        if word_clean in terms_dict:
            found_terms.append(word_clean)
            found_scores.append(terms_dict[word_clean])

    count = len(found_terms)
    # weighted average score
    avg_score = round(sum(found_scores) / count, fpdigits) if count > 0 else 0.0

    unique_count = len(set(found_terms))

    return unique_count, avg_score


def _count_questions(text: str) -> int:
    """Count question marks in text."""
    return text.count("?")


async def _compare_profiles(
    real_profile: dict,
    simulated_profile: dict,
    llm: LLM,
) -> Dict[str, int]:
    """Compare real and simulated patient profiles and categorize differences."""
    fidelity_counts = {
        "Supported": 0,
        "Implied": 0,
        "Contradiction": 0,
        "Hallucination": 0,
        "Not Discussed": 0,
    }

    for key in real_profile.keys():
        if real_profile[key] == "Unknown":
            continue
        real_items = [item.strip() for item in real_profile[key].split(";")]
        simulated_items = [item.strip() for item in simulated_profile[key].split(";")]
        for sim_item in simulated_items:
            if sim_item in real_items:
                fidelity_counts["Supported"] += 1
            elif sim_item == "Unknown" and real_items == ["Unknown"]:
                fidelity_counts["Supported"] += 1
            elif sim_item == "Unknown" and real_items != ["Unknown"]:
                fidelity_counts["Not Discussed"] += 1
            else:
                prompt = PROFILE_FIDELITY_PROMPT.format(
                    real_item="; ".join(real_items),
                    simulated_item=sim_item,
                )
                response = await llm.generate_response(prompt=prompt)
                category = extract_tagged_text(response["response"], "category")
                if category in fidelity_counts:
                    fidelity_counts[category] += 1

    return fidelity_counts


async def evaluate_profile_fidelity(
    full_conv: pd.DataFrame,
    case_data: Dict[str, Any],
    llm: LLM,
    extraction_prompt: str = EVAL_CASE_EXTRACTION_PROMPT,
    patient_profile_class=CaseDescription,
) -> Dict[str, int]:
    transcript_lines = []
    for _, row in full_conv.iterrows():
        if pd.notna(row["doctor_question"]):
            transcript_lines.append(f"Doctor: {row['doctor_question']}")
        if pd.notna(row["simulated_response"]):
            transcript_lines.append(f"Patient: {row['simulated_response']}")
    transcript = "\n".join(transcript_lines)

    real_profile = case_data

    simulated_profile = await extract_patient_profile_from_text(
        transcript, llm, extraction_prompt, patient_profile_class
    )

    for key in real_profile:
        if key not in simulated_profile:
            simulated_profile[key] = "Unknown"

    return await _compare_profiles(
        real_profile=real_profile,
        simulated_profile=simulated_profile,
        llm=llm,
    ), simulated_profile


async def evaluate_fact_fidelity(
    patient_turns: list[str],
    personal_fields: str,
    case_data: Dict[str, Any],
    llm: LLM,
) -> tuple[dict[str, int], dict[str, Any]]:
    """Evaluate fact-level fidelity using one extraction call and one batch judging call."""

    extraction_prompt = f"""You are a specialized medical scribe assistant. Your task is to extract all health related facts from the following patient answers.

    ### Guidelines for Extraction:

    1.  **Strict Adherence to Facts:** Extract ONLY information explicitly stated in the transcript. Do not infer diagnoses, assume unstated details, or fill gaps with general knowledge. If a specific detail is not spoken, return "Unknown".
    2.  **Neutral & Objective Tone:** Use a detached, clinical tone. Avoid emotive language, subjective interpretations, or judgment.
    3.  **Telegraphic Style:**
        * Eliminate pronouns (I, he, she), articles (a, an, the), and unnecessary verbs.
        * Use noun phrases and standard medical terminology where applicable (e.g., replace "trouble breathing" with "dyspnea" only if clinically certain; otherwise use exact descriptive phrasing like "shortness of breath").
    4.  **Granularity & Detail:**
        * Capture specific modifiers: severity, duration, frequency, dosage, location, and radiation.
        * *Example:* Instead of "Headache", use "pulsating frontal headache; 3-day duration; 7/10 severity".
    5.  **Formatting:**
        * Output strictly valid list of strings.

    Patient turns:\n{"; ".join(patient_turns)}\n\n"""

    extracted = await llm.generate_response(
        prompt=extraction_prompt,
        outlines_class=_AtomicFactsExtraction,
    )
    parsed = extracted["response"]
    payload = (
        parsed.model_dump()
        if isinstance(parsed, pydantic.BaseModel)
        else _AtomicFactsExtraction.model_validate(parsed).model_dump()
    )
    atomic_facts = [str(fact).strip() for fact in payload["facts"] if str(fact).strip()]
    case_fields_json = json.dumps(case_data, ensure_ascii=False, indent=2)

    print("---")
    print(f"EXTRACTED FACTS: {atomic_facts}")
    print(f"PERSONAL FIELDS: {personal_fields}")
    print(f"GIVEN CASE: {case_fields_json}")

    fidelity_counts = {
        "Supported": 0,
        "Hallucination": 0,
    }
    if not atomic_facts:
        details = {
            "extracted_facts": [],
            "fact_verdicts": {},
            "input_token_count": int(extracted["prompt_token_count"]),
            "output_token_count": int(extracted["output_token_count"]),
        }
        return fidelity_counts, details

    facts_json = json.dumps(atomic_facts, ensure_ascii=False, indent=2)
    prompt = FACT_FIDELITY_PROMPT.format(
        personal_fields=personal_fields,
        case_fields=case_fields_json,
        facts=facts_json,
    )
    response = await llm.generate_response(
        prompt=prompt,
        outlines_class=_FactFidelityBatch,
    )

    parsed = response["response"]
    payload = (
        parsed.model_dump()
        if isinstance(parsed, pydantic.BaseModel)
        else _FactFidelityBatch.model_validate(parsed).model_dump()
    )
    supported_flags = [bool(flag) for flag in payload["supported"]]
    if len(supported_flags) != len(atomic_facts):
        raise ValueError(
            "Fact fidelity output length mismatch: "
            f"expected {len(atomic_facts)}, got {len(supported_flags)}"
        )

    fact_verdicts = {
        fact: ("Supported" if is_supported else "Hallucination")
        for fact, is_supported in zip(atomic_facts, supported_flags)
    }

    for verdict in fact_verdicts.values():
        if verdict == "Supported":
            fidelity_counts["Supported"] += 1
        else:
            fidelity_counts["Hallucination"] += 1

    details = {
        "extracted_facts": atomic_facts,
        "fact_verdicts": fact_verdicts,
        "input_token_count": int(extracted["prompt_token_count"])
        + int(response["prompt_token_count"]),
        "output_token_count": int(extracted["output_token_count"])
        + int(response["output_token_count"]),
    }

    print("---")
    print(f"FACT VERDICTS: {fact_verdicts}")

    print("---")
    print("Fact-level fidelity counts:", fidelity_counts)
    return fidelity_counts, details


def _extract_personal_information_pairs(personal_fields: str) -> list[tuple[str, str]]:
    """Extract ordered personal information key-value pairs from x_input text."""
    lines = [line.strip() for line in str(personal_fields).splitlines()]
    in_personal_section = False
    pairs: list[tuple[str, str]] = []

    for line in lines:
        if not line:
            continue
        lower = line.lower()
        if lower == "personal information:":
            in_personal_section = True
            continue
        if lower == "hexaco personality:":
            break
        if not in_personal_section:
            continue
        if ":" not in line:
            continue

        field, value = line.split(":", 1)
        field_clean = field.strip()
        value_clean = value.strip()
        if not field_clean or not value_clean:
            continue
        if value_clean.lower() == "unknown":
            continue
        pairs.append((field_clean, value_clean))

    return pairs


async def evaluate_personal_information_presence(
    personal_fields: str,
    latent_role: str,
    llm: LLM,
) -> tuple[dict[str, int], dict[str, Any]]:
    """Evaluate if personal information fields are contained in a latent role description."""
    personal_info_pairs = _extract_personal_information_pairs(personal_fields)

    containment_counts = {
        "Contained": 0,
        "Missing": 0,
    }
    if not personal_info_pairs:
        details = {
            "personal_info_fields": [],
            "personal_info_presence": {},
            "input_token_count": 0,
            "output_token_count": 0,
        }
        return containment_counts, details

    personal_info_payload = [
        {"field": field, "value": value} for field, value in personal_info_pairs
    ]
    personal_info_json = json.dumps(personal_info_payload, ensure_ascii=False, indent=2)

    prompt = PERSONAL_INFO_PRESENCE_PROMPT.format(
        personal_info_fields=personal_info_json,
        latent_role=latent_role,
    )
    response = await llm.generate_response(
        prompt=prompt,
        outlines_class=_PersonalInfoPresenceBatch,
    )

    parsed = response["response"]
    payload = (
        parsed.model_dump()
        if isinstance(parsed, pydantic.BaseModel)
        else _PersonalInfoPresenceBatch.model_validate(parsed).model_dump()
    )
    contained_flags = [bool(flag) for flag in payload["contained"]]
    if len(contained_flags) != len(personal_info_pairs):
        raise ValueError(
            "Personal information presence output length mismatch: "
            f"expected {len(personal_info_pairs)}, got {len(contained_flags)}"
        )

    presence_by_field = {
        field: {
            "value": value,
            "contained": is_contained,
        }
        for (field, value), is_contained in zip(personal_info_pairs, contained_flags)
    }

    for is_contained in contained_flags:
        if is_contained:
            containment_counts["Contained"] += 1
        else:
            containment_counts["Missing"] += 1

    details = {
        "personal_info_fields": personal_info_payload,
        "personal_info_presence": presence_by_field,
        "input_token_count": int(response["prompt_token_count"]),
        "output_token_count": int(response["output_token_count"]),
    }

    return containment_counts, details


async def evaluate_response_quality(
    doctor_question: str,
    patient_response: str,
    llm: LLM,
    real_response: Optional[str] = None,
    evaluation_type: str = "relevance",
) -> str:
    """Evaluate patient response quality using LLM-as-a-Judge."""
    if evaluation_type == "relevance":
        if patient_response.strip() == "":
            return "Irrelevant/Evasive"
        prompt = RELEVANCE_EVALUATION_PROMPT.format(
            doctor_question=doctor_question,
            patient_response=patient_response,
        )

    elif evaluation_type == "realism_content":
        if not real_response:
            return "Dissimilar"
        prompt = REALISM_CONTENT_EVALUATION_PROMPT.format(
            real_response=real_response,
            patient_response=patient_response,
        )

    elif evaluation_type == "realism_style":
        if not real_response:
            return "Dissimilar"
        prompt = REALISM_STYLE_EVALUATION_PROMPT.format(
            real_response=real_response,
            patient_response=patient_response,
        )

    else:
        raise ValueError(f"Invalid evaluation_type: {evaluation_type}")

    resp = await llm.generate_response(prompt=prompt)
    return (
        extract_tagged_text(resp["response"], "category")
        .replace('"', "")
        .replace("'", "")
    )


async def evaluate_persona_consistency(
    first_responses: str,
    last_responses: str,
    llm: LLM,
) -> str:
    """Evaluate persona consistency between first and last patient responses using LLM-as-a-Judge."""
    prompt = PERSONA_CONSISTENCY_EVALUATION_PROMPT.format(
        first_responses=first_responses,
        last_responses=last_responses,
    )
    resp = await llm.generate_response(prompt=prompt)
    return (
        extract_tagged_text(resp["response"], "category")
        .replace('"', "")
        .replace("'", "")
    )


async def reconstruct_personality_params(
    conversation: str,
    llm: LLM,
    subset: Optional[set[str]] = None,
) -> Dict[str, Optional[int]]:
    """Reconstruct requested HEXACO axis levels using one structured LLM call."""
    trait_prompts = {
        "H": PERSONALITY_RECON_H_PROMPT,
        "E": PERSONALITY_RECON_E_PROMPT,
        "X": PERSONALITY_RECON_X_PROMPT,
        "A": PERSONALITY_RECON_A_PROMPT,
        "C": PERSONALITY_RECON_C_PROMPT,
        "O": PERSONALITY_RECON_O_PROMPT,
    }
    axis_order = ["H", "E", "X", "A", "C", "O"]
    subset_upper = {axis.upper() for axis in subset} if subset is not None else None
    requested_axes = [
        axis for axis in axis_order if subset_upper is None or axis in subset_upper
    ]
    if not requested_axes:
        raise ValueError("No valid HEXACO axes requested for reconstruction.")

    axis_prompts = "\n\n".join(trait_prompts[axis].strip() for axis in requested_axes)

    schema_fields = {
        axis: (
            pydantic.conint(ge=1, le=3),
            pydantic.Field(description=f"{axis} level as integer 1..3"),
        )
        for axis in requested_axes
    }
    ReconSchema = pydantic.create_model("HexacoReconSchema", **schema_fields)  # type: ignore

    prompt = (
        "You are an expert judge inferring HEXACO axes from a doctor-patient conversation.\n"
        "Use only patient behavior in the provided conversation and choose one level (1..3) per requested axis.\n"
        "Return only the structured output for requested axes.\n\n"
        f"Requested axes: {', '.join(requested_axes)}\n\n"
        "Axis definitions and level criteria:\n"
        f"{axis_prompts}\n\n"
        f"Conversation:\n{conversation}"
    )
    response = await llm.generate_response(prompt=prompt, outlines_class=ReconSchema)
    parsed = response["response"]
    payload = (
        parsed.model_dump() if isinstance(parsed, pydantic.BaseModel) else dict(parsed)
    )

    print("---")
    print("RECON CONV:", conversation)
    print("RECON RESPONSE:", response)

    return {axis: int(payload[axis]) for axis in requested_axes}


def _token_ratio(base: str, compare: str, fpdigits: int = 3) -> float:
    """Compute token ratio (compare / base)."""
    base_len = len(base.split())
    compare_len = len(compare.split())
    return round(compare_len / base_len, fpdigits) if base_len > 0 else 0.0


def _available_case_fields(case_description: Dict[str, Any]) -> list[str]:
    if not isinstance(case_description, dict):
        raise ValueError("case_description must be a dict for info-control metrics")
    return [f for f in LEISURE_FIELDS + MEDICAL_FIELDS if f in case_description]


async def classify_doctor_question_fields(
    question: str,
    available_fields: list[str],
    llm: LLM,
) -> list[str]:
    if not question.strip() or not available_fields:
        return []
    prompt = PWP_CLASS.format(
        available_fields=", ".join(available_fields), question=question
    )
    response = await llm.generate_response(prompt=prompt, outlines_class=StringFields)
    parsed = response["response"]
    payload = (
        parsed.model_dump()
        if isinstance(parsed, pydantic.BaseModel)
        else StringFields.model_validate(parsed).model_dump()
    )
    return [f for f in payload["fields"] if f in available_fields]


async def extract_facts_with_fields(
    patient_turn: str,
    available_fields: list[str],
    case_description: Dict[str, Any],
    llm: LLM,
) -> list[dict]:
    if not patient_turn.strip():
        return []
    case_subset = {f: case_description.get(f, "Unknown") for f in available_fields}
    prompt = TAGGED_FACT_EXTRACTION_PROMPT.format(
        available_fields=", ".join(available_fields + ["other"]),
        case_fields=json.dumps(case_subset, ensure_ascii=False, indent=2),
        patient_turn=patient_turn,
    )
    response = await llm.generate_response(
        prompt=prompt, outlines_class=_TaggedFactsExtraction
    )
    parsed = response["response"]
    payload = (
        parsed.model_dump()
        if isinstance(parsed, pydantic.BaseModel)
        else _TaggedFactsExtraction.model_validate(parsed).model_dump()
    )
    allowed = set(available_fields) | {"other"}
    cleaned = []
    for fact in payload["facts"]:
        field = fact["field"] if fact["field"] in allowed else "other"
        cleaned.append(
            {
                "text": str(fact["text"]).strip(),
                "field": field,
                "supported": bool(fact["supported"]),
            }
        )
    return cleaned


def compute_time_to_fact(
    turn_fact_maps: list[list[dict]], case_fields: list[str]
) -> Dict[str, Any]:
    if not case_fields:
        return {"ttf_per_field": {}, "median_ttf": None, "disclosure_auc": None}

    ttf_per_field: Dict[str, Optional[int]] = {f: None for f in case_fields}
    for turn_idx, facts in enumerate(turn_fact_maps, start=1):
        for fact in facts:
            field = fact["field"]
            if (
                field in ttf_per_field
                and ttf_per_field[field] is None
                and fact["supported"]
            ):
                ttf_per_field[field] = turn_idx

    total_turns = len(turn_fact_maps)
    disclosed_ttfs = [t for t in ttf_per_field.values() if t is not None]
    median_ttf = float(pd.Series(disclosed_ttfs).median()) if disclosed_ttfs else None

    if total_turns == 0:
        disclosure_auc = None
    else:
        fractions = []
        for t in range(1, total_turns + 1):
            disclosed = sum(
                1 for ttf in ttf_per_field.values() if ttf is not None and ttf <= t
            )
            fractions.append(disclosed / len(case_fields))
        disclosure_auc = round(sum(fractions) / total_turns, 4)

    return {
        "ttf_per_field": ttf_per_field,
        "median_ttf": median_ttf,
        "disclosure_auc": disclosure_auc,
    }


def compute_prompted_disclosure(
    turn_fact_maps: list[list[dict]],
    requested_fields_per_turn: list[list[str]],
) -> Tuple[list[dict], Dict[str, Any]]:
    per_turn: list[dict] = []
    for facts, requested in zip(turn_fact_maps, requested_fields_per_turn):
        mentioned = {f["field"] for f in facts if f["field"] != "other"}
        requested_set = set(requested)
        if not mentioned and not requested_set:
            per_turn.append({"pdp_precision": None, "pdp_recall": None, "pdp_f1": None})
            continue
        overlap = len(mentioned & requested_set)
        precision = overlap / len(mentioned) if mentioned else 0.0
        recall = overlap / len(requested_set) if requested_set else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )
        per_turn.append(
            {
                "pdp_precision": round(precision, 4),
                "pdp_recall": round(recall, 4),
                "pdp_f1": round(f1, 4),
            }
        )

    def _mean(key: str) -> Optional[float]:
        vals = [row[key] for row in per_turn if row[key] is not None]
        return round(sum(vals) / len(vals), 4) if vals else None

    aggregate = {
        "pdp_precision_mean": _mean("pdp_precision"),
        "pdp_recall_mean": _mean("pdp_recall"),
        "pdp_f1_mean": _mean("pdp_f1"),
    }
    return per_turn, aggregate


def compute_unprompted_leakage(
    turn_fact_maps: list[list[dict]],
    requested_fields_per_turn: list[list[str]],
) -> Dict[str, Any]:
    ever_requested: set[str] = set()
    prompted = 0
    unprompted = 0
    for facts, requested in zip(turn_fact_maps, requested_fields_per_turn):
        ever_requested.update(requested)
        for fact in facts:
            if not fact["supported"]:
                continue
            if fact["field"] == "other":
                unprompted += 1
                continue
            if fact["field"] in ever_requested:
                prompted += 1
            else:
                unprompted += 1
    total = prompted + unprompted
    rate = round(unprompted / total, 4) if total > 0 else None
    return {
        "unprompted_leakage_rate": rate,
        "unprompted_fact_count": unprompted,
        "prompted_fact_count": prompted,
    }


async def evaluate_realism_judge(
    full_conv: pd.DataFrame,
    llm: LLM,
) -> Dict[str, Any]:
    transcript_lines = []
    for _, row in full_conv.iterrows():
        if pd.notna(row["doctor_question"]):
            transcript_lines.append(f"Doctor: {row['doctor_question']}")
        if pd.notna(row["simulated_response"]):
            transcript_lines.append(f"Patient: {row['simulated_response']}")
    transcript = "\n".join(transcript_lines)

    prompt = REALISM_JUDGE_PROMPT.format(transcript=transcript)
    response = await llm.generate_response(
        prompt=prompt, outlines_class=_RealismJudgeOutput
    )
    parsed = response["response"]
    payload = (
        parsed.model_dump()
        if isinstance(parsed, pydantic.BaseModel)
        else _RealismJudgeOutput.model_validate(parsed).model_dump()
    )
    return {
        "judge_p_real": float(payload["p_real"]),
        "judge_classification": "real" if payload["p_real"] >= 0.5 else "simulated",
        "judge_symptom_realism": int(payload["symptom_realism"]),
        "judge_information_control": int(payload["information_control"]),
        "judge_style_realism": int(payload["style_realism"]),
        "judge_justification": payload["justification"],
    }


async def evaluate_conversation(
    full_conv: pd.DataFrame,
    llm: LLM,
    case_description: str | dict = None,
    metrics: Optional[List[str]] = None,
) -> tuple[pd.DataFrame, Dict[str, Any]]:
    """Evaluate a pre-simulated conversation DataFrame."""

    # Define a default of metrics to do if none are provided
    if metrics is None:
        metrics = [
            "relevance",
            "realism_content",
            "realism_style",
            "token_ratios",
            "token_count",
            "question_count",
            "domain_terms",
            "sentiment_score",
            "readability",
            "lexical_diversity",
            "persona_consistency",
            "profile_fidelity",
            "personality_reconstruction",
            "time_to_fact",
            "prompted_disclosure",
            "unprompted_leakage",
        ]

    full_conv = full_conv.copy()
    llm_judge_evals = ["relevance", "realism_content", "realism_style"]
    info_control_metrics = {"time_to_fact", "prompted_disclosure", "unprompted_leakage"}
    needs_primitives = bool(info_control_metrics & set(metrics))

    if needs_primitives and not isinstance(case_description, dict):
        raise ValueError(
            "time_to_fact / prompted_disclosure / unprompted_leakage require a dict case_description"
        )

    available_fields = (
        _available_case_fields(case_description) if needs_primitives else []
    )

    terms_dict = _load_domain_terms() if "domain_terms" in metrics else None
    sentiment_analyzer = (
        SentimentIntensityAnalyzer() if "sentiment_score" in metrics else None
    )

    for idx, row in tqdm(full_conv.iterrows(), total=len(full_conv), leave=False):
        doctor_q = (
            str(row["doctor_question"]) if pd.notna(row["doctor_question"]) else ""
        )
        real_resp = str(row["real_response"]) if pd.notna(row["real_response"]) else ""
        simulated_resp = (
            str(row["simulated_response"])
            if pd.notna(row["simulated_response"])
            else ""
        )

        for metric in llm_judge_evals:
            if metric in metrics:
                eval_result = await evaluate_response_quality(
                    doctor_question=doctor_q,
                    patient_response=simulated_resp,
                    llm=llm,
                    real_response=real_resp,
                    evaluation_type=metric,
                )
                full_conv.at[idx, f"{metric}_category"] = eval_result

        if "token_ratios" in metrics:
            full_conv.at[idx, "token_ratio_patient_real_sim"] = _token_ratio(
                real_resp, simulated_resp
            )
            full_conv.at[idx, "token_ratio_doctor_patient_real"] = _token_ratio(
                doctor_q, real_resp
            )
            full_conv.at[idx, "token_ratio_doctor_patient_sim"] = _token_ratio(
                doctor_q, simulated_resp
            )

        if "sentiment_score" in metrics and sentiment_analyzer:
            real_sentiment = sentiment_analyzer.polarity_scores(real_resp)
            sim_sentiment = sentiment_analyzer.polarity_scores(simulated_resp)
            for key in ["neg", "neu", "pos"]:
                full_conv.at[idx, f"sentiment_{key}_real"] = real_sentiment[key]
                full_conv.at[idx, f"sentiment_{key}_sim"] = sim_sentiment[key]

        if needs_primitives:
            requested = await classify_doctor_question_fields(
                doctor_q, available_fields, llm
            )
            turn_facts = await extract_facts_with_fields(
                simulated_resp, available_fields, case_description, llm
            )
            full_conv.at[idx, "requested_fields"] = json.dumps(requested)
            full_conv.at[idx, "turn_fact_map"] = json.dumps(turn_facts)

    full_real_text = "\n".join(full_conv["real_response"].fillna("").astype(str))
    full_sim_text = "\n".join(full_conv["simulated_response"].fillna("").astype(str))

    # Aggregate metrics
    def _count_categories(series: pd.Series) -> Dict[str, int]:
        """Count occurrences of each category."""
        return series.value_counts().to_dict()

    def _perc_category(series: pd.Series, category: str) -> float:
        """Compute percentage of a specific category."""
        total = len(series)
        if total == 0:
            return 0.0
        count = (series == category).sum()
        return float(count) / total

    def _mean_series(series: pd.Series, fpdigits: int = 3) -> float:
        vals = series.dropna().astype(float)
        return round(float(vals.mean()), fpdigits)

    def _r_score(text: str) -> Optional[float]:
        return (
            Readability(text).flesch_kincaid().score
            if len(text.split()) > 100
            else None
        )

    def _type_token_ratio(text: str, fpdigits: int = 3) -> float:
        words = text.split()
        if not words:
            return 0.0
        unique_words = set(words)
        ttr = len(unique_words) / len(words)
        return round(ttr, fpdigits)

    aggregate: Dict[str, Any] = {"total_steps": int(len(full_conv))}

    for metric in [m for m in llm_judge_evals if m in metrics]:
        category_col = f"{metric}_category"
        if metric.startswith("realism_"):
            aggregate[f"{metric}_similarity"] = _perc_category(
                full_conv[category_col], "Similar"
            )
        else:
            aggregate[f"{metric}_distribution"] = _count_categories(
                full_conv[category_col]
            )

    # Evaluate Persona Consistency
    if "persona_consistency" in metrics:
        n_responses = 5
        if len(full_conv) < n_responses * 2:
            aggregate["persona_consistency"] = "Insufficient Data"
            aggregate["persona_consistency_explanation"] = (
                f"Not enough responses to evaluate persona consistency "
                f"(need at least {n_responses * 2}, got {len(full_conv)})"
            )
        else:
            first_responses = "; ".join(
                full_conv["simulated_response"].head(n_responses).fillna("").astype(str)
            )
            last_responses = "; ".join(
                full_conv["simulated_response"].tail(n_responses).fillna("").astype(str)
            )

            persona_eval = await evaluate_persona_consistency(
                first_responses=first_responses,
                last_responses=last_responses,
                llm=llm,
            )

            aggregate["persona_consistency"] = persona_eval

    # Reconstruct personality parameters based on full conversation
    if "personality_reconstruction" in metrics:
        try:
            reconstruction_transcript_lines = []
            for _, row in full_conv.iterrows():
                if pd.notna(row["doctor_question"]):
                    reconstruction_transcript_lines.append(
                        f"Doctor: {row['doctor_question']}"
                    )
                if pd.notna(row["simulated_response"]):
                    reconstruction_transcript_lines.append(
                        f"Patient: {row['simulated_response']}"
                    )
            reconstruction_transcript = "\n".join(reconstruction_transcript_lines)

            reconstruction = await reconstruct_personality_params(
                conversation=reconstruction_transcript,
                llm=llm,
            )
            for key, value in reconstruction.items():
                aggregate[f"personality_reconstructed_{key}"] = value
        except Exception as e:
            logging.error(
                f"Personality reconstruction failed with error: {type(e).__name__}: {str(e)}"
            )
            logging.error("Full traceback:", exc_info=True)

            for key in [
                "honesty",
                "emotional_state",
                "extraversion",
                "agreeableness",
                "conscientiousness",
                "openness",
            ]:
                aggregate[f"personality_reconstructed_{key}"] = None

    # Evaluate Profile Fidelity
    if "profile_fidelity" in metrics and case_description is not None:
        fidelity_distribution, sim_profile = await evaluate_profile_fidelity(
            full_conv=full_conv,
            case_data=case_description,
            llm=llm,
        )
        aggregate["profile_fidelity_distribution"] = fidelity_distribution
    else:
        sim_profile = {}
        aggregate["profile_fidelity_distribution"] = "No case description provided"

    # Aggregate token ratio
    if "token_ratios" in metrics:
        aggregate["mean_token_ratio_patient_real_sim"] = _mean_series(
            full_conv["token_ratio_patient_real_sim"]
        )
        aggregate["mean_token_ratio_doctor_patient_real"] = _mean_series(
            full_conv["token_ratio_doctor_patient_real"]
        )
        aggregate["mean_token_ratio_doctor_patient_sim"] = _mean_series(
            full_conv["token_ratio_doctor_patient_sim"]
        )

    if "token_count" in metrics:
        aggregate["mean_token_count_doctor"] = _mean_series(
            full_conv["doctor_question"]
            .fillna("")
            .astype(str)
            .map(lambda t: len(t.split()))
        )
        aggregate["mean_token_count_patient_sim"] = _mean_series(
            full_conv["simulated_response"]
            .fillna("")
            .astype(str)
            .map(lambda t: len(t.split()))
        )
        aggregate["mean_token_count_patient_real"] = _mean_series(
            full_conv["real_response"]
            .fillna("")
            .astype(str)
            .map(lambda t: len(t.split()))
        )

    # Aggregate question counts over full transcripts
    if "question_count" in metrics:
        aggregate["question_count_real"] = _count_questions(full_real_text)
        aggregate["question_count_simulated"] = _count_questions(full_sim_text)

    # Aggregate domain term statistics over full transcripts
    if "domain_terms" in metrics and terms_dict is not None:
        real_count, real_score = _count_domain_terms(full_real_text, terms_dict)
        sim_count, sim_score = _count_domain_terms(full_sim_text, terms_dict)
        aggregate["domain_term_count_real"] = int(real_count)
        aggregate["domain_term_count_sim"] = int(sim_count)
        aggregate["domain_term_avg_score_real"] = real_score
        aggregate["domain_term_avg_score_sim"] = sim_score
        if real_count > 0:
            aggregate["domain_term_ratio"] = round(sim_count / real_count, 3)
        else:
            aggregate["domain_term_ratio"] = 0.0

    if "sentiment_score" in metrics:
        for key in ["neg", "neu", "pos"]:
            aggregate[f"mean_sentiment_{key}_real"] = _mean_series(
                full_conv[f"sentiment_{key}_real"]
            )
            aggregate[f"mean_sentiment_{key}_sim"] = _mean_series(
                full_conv[f"sentiment_{key}_sim"]
            )

    if "readability" in metrics:
        aggregate["readability_score_real"] = _r_score(full_real_text)
        aggregate["readability_score_sim"] = _r_score(full_sim_text)

    if "lexical_diversity" in metrics:
        aggregate["lexical_diversity_real"] = _type_token_ratio(full_real_text)
        aggregate["lexical_diversity_sim"] = _type_token_ratio(full_sim_text)

    if needs_primitives:
        turn_fact_maps = [
            json.loads(full_conv.at[idx, "turn_fact_map"]) for idx in full_conv.index
        ]
        requested_fields_per_turn = [
            json.loads(full_conv.at[idx, "requested_fields"]) for idx in full_conv.index
        ]

        if "time_to_fact" in metrics:
            ttf_result = compute_time_to_fact(turn_fact_maps, available_fields)
            aggregate["median_ttf"] = ttf_result["median_ttf"]
            aggregate["disclosure_auc"] = ttf_result["disclosure_auc"]
            aggregate["ttf_per_field"] = ttf_result["ttf_per_field"]

        if "prompted_disclosure" in metrics:
            per_turn_pdp, pdp_agg = compute_prompted_disclosure(
                turn_fact_maps, requested_fields_per_turn
            )
            for idx, pdp_row in zip(full_conv.index, per_turn_pdp):
                for key, value in pdp_row.items():
                    full_conv.at[idx, key] = value
            aggregate.update(pdp_agg)

        if "unprompted_leakage" in metrics:
            aggregate.update(
                compute_unprompted_leakage(turn_fact_maps, requested_fields_per_turn)
            )

    if "realism_judge" in metrics:
        aggregate.update(await evaluate_realism_judge(full_conv=full_conv, llm=llm))

    return full_conv, aggregate, sim_profile
