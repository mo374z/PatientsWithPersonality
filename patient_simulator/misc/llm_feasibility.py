import ast
import json
import re
from pathlib import Path
from typing import Any

import pydantic
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from patient_simulator.misc.llm import LLM
from patient_simulator.prompts.patient_prompts import (
    CRAFTMD_SYS,
    PWP_LATENT_ROLE,
    PWP_A,
    PWP_C,
    PWP_CLASS,
    PWP_E,
    PWP_E_META,
    PWP_H,
    PWP_H_DOWNPLAY,
    PWP_O,
    PWP_O_META,
    PWP_SYS,
    PWP_X,
)
from patient_simulator.misc.utils import extract_tagged_text


AVAILABLE_FIELDS = [
    "tobacco",
    "alcohol",
    "illicit_drug",
    "sexual_history",
    "exercise",
    "allergies",
    "family_medical_history",
    "medical_device",
    "medical_history",
    "present_illness_positive",
    "present_illness_negative",
    "pain",
    "medication",
]

PERSONAL_FIELDS = [
    "age",
    "gender",
    "marital_status",
    "children",
    "living_situation",
    "occupation",
    "insurance",
    "arrival_transport",
    "chiefcomplaint",
]


class StringFields(pydantic.BaseModel):
    """Schema for a list of fields."""

    fields: list[str]


def tokenize_words(text: str) -> list[str]:
    """Split text into lowercase word tokens."""
    return re.findall(r"[a-zA-Z']+", (text or "").lower())


def jaccard_similarity(words_a: list[str], words_b: list[str]) -> float:
    """Compute Jaccard similarity between two word lists."""
    set_a = set(words_a)
    set_b = set(words_b)
    if not set_a and not set_b:
        return 1.0
    union = set_a | set_b
    if not union:
        return 0.0
    return len(set_a & set_b) / len(union)


def has_emotion_signal(
    text: str, words: list[str], analyzer: SentimentIntensityAnalyzer
) -> bool:
    """Detect emotional expression using stem matches and sentiment intensity."""
    emotion_stems = {
        "anx",
        "worr",
        "fear",
        "afraid",
        "panic",
        "distress",
        "irrit",
        "ang",
        "frustr",
        "upset",
        "overwhelm",
        "detach",
        "withdraw",
        "tense",
        "nerv",
        "defens",
        "catastroph",
        "hopeless",
        "helpless",
    }

    stem_match = any(any(stem in word for stem in emotion_stems) for word in words)
    if stem_match:
        return True

    compound = analyzer.polarity_scores(text).get("compound", 0.0)
    return abs(compound) >= 0.2


def contains_number(text: str) -> bool:
    """Detect whether text contains any digit."""
    return re.search(r"\d", text or "") is not None


def format_case_description(case_description: dict[str, Any]) -> str:
    """Format a case description dictionary for prompt insertion."""
    return "\n".join([f"    {key}: {value}" for key, value in case_description.items()])


def load_case_description(
    case_name: str = "CAR0001", data_root: str | Path = "data/extracted_profiles"
) -> dict[str, Any]:
    """Load a patient case description JSON file."""
    file_path = Path(data_root) / f"{case_name}.json"
    with file_path.open("r") as file:
        return json.load(file)


def parse_fields_response(raw_response: Any) -> list[str]:
    """Parse field names from structured or text response formats."""
    if isinstance(raw_response, StringFields):
        return raw_response.fields
    if isinstance(raw_response, dict) and "fields" in raw_response:
        fields = raw_response["fields"]
        return fields if isinstance(fields, list) else []
    if isinstance(raw_response, list):
        return [str(item) for item in raw_response]
    if isinstance(raw_response, str):
        text = raw_response.strip()
        if not text:
            return []
        if text.lower() in {"none", "[]"}:
            return []
        try:
            parsed = ast.literal_eval(text)
            if isinstance(parsed, dict) and "fields" in parsed:
                return [str(item) for item in parsed.get("fields", [])]
            if isinstance(parsed, list):
                return [str(item) for item in parsed]
        except (ValueError, SyntaxError):
            pass
        return [token.strip() for token in text.split(",") if token.strip()]
    return []


async def build_everyday_system_prompt(
    case_description: dict[str, Any],
    meta_llm: LLM,
    agreeableness_level: int,
    openness_level: int,
    emotionality_level: int = 1,
    honesty_level: int = 1,
    extraversion_level: int = 1,
    conscientiousness_level: int = 1,
    level: str = "B",
    tangent_topic: str = "",
    prior_belief: str = "",
) -> str:
    """Build PWP_SYS by generating a latent role from personal info + HEXACO."""
    personal_information = "\n".join(
        [
            f"\t{field}: {case_description[field]}"
            for field in PERSONAL_FIELDS
            if field in case_description
            and str(case_description[field]).strip().lower() != "unknown"
        ]
    )
    personal_information += f"\n\tcefr_level: {level}"
    if tangent_topic:
        personal_information += f"\n\tfurther information: {tangent_topic}"

    if emotionality_level == 3:
        emotion_response = await meta_llm.generate_response(
            prompt=PWP_E_META.format(personal_information=personal_information)
        )
        emotionality = extract_tagged_text(str(emotion_response["response"]), "state")
    else:
        emotionality = PWP_E[emotionality_level]

    if openness_level == 3 and not prior_belief:
        prior_response = await meta_llm.generate_response(
            prompt=PWP_O_META.format(chiefcomplaint=case_description["chiefcomplaint"])
        )
        prior_belief = extract_tagged_text(
            str(prior_response["response"]), "prior_belief"
        )

    openness = (
        PWP_O[3].format(prior_belief=prior_belief)
        if openness_level == 3
        else PWP_O[openness_level]
    )

    hexaco = "\n".join(
        [
            f"\tHonesty-Humility: {PWP_H[honesty_level]}",
            f"\tEmotionality: {emotionality}",
            f"\tExtraversion: {PWP_X[extraversion_level]}",
            f"\tAgreeableness: {PWP_A[agreeableness_level]}",
            f"\tConscientiousness: {PWP_C[conscientiousness_level]}",
            f"\tOpenness: {openness}",
        ]
    )

    latent_response = await meta_llm.generate_response(
        prompt=PWP_LATENT_ROLE.format(
            personal_information=personal_information,
            hexaco_personality=hexaco,
        )
    )
    latent_role = extract_tagged_text(str(latent_response["response"]), "role")

    structured_information = "\n".join(
        [
            f"    {field}: {case_description[field]}"
            for field in AVAILABLE_FIELDS
            if field in case_description
        ]
    )

    return PWP_SYS.format(
        latent_role=latent_role,
        structured_information=structured_information,
    )


async def test_downplay(meta_llm: LLM) -> tuple[float, list[str]]:
    """Evaluate whether downplay phrasing follows safety and style constraints."""
    examples = [
        "drinks a bottle of vodka every day.",
        "pain is unbearable, 10/10.",
        "can't breathe at all.",
    ]
    severe_words = {
        "unbearable",
        "severe",
        "critical",
        "fatal",
        "dying",
        "breathe",
        "emergency",
        "intense",
        "extreme",
    }

    successful = 0
    responses: list[str] = []
    for example in examples:
        prompt = PWP_H_DOWNPLAY.format(leisure_info=example)
        response = await meta_llm.generate_response(prompt=prompt)
        phrase = extract_tagged_text(str(response["response"]), "phrase")
        responses.append(phrase)
        words = tokenize_words(phrase)
        length_ok = len(words) <= 10
        number_ok = not contains_number(phrase)
        severe_ok = not any(word in severe_words for word in words)
        if length_ok and number_ok and severe_ok:
            successful += 1

    return successful / len(examples), responses


async def test_emotional_state(meta_llm: LLM) -> tuple[float, list[str]]:
    """Evaluate novelty and emotional content of generated distress states."""
    prompt = PWP_E_META.format(
        personal_information="age: 42\nchiefcomplaint: chest pain\ncefr_level: B"
    )
    analyzer = SentimentIntensityAnalyzer()

    outputs: list[str] = []
    for _ in range(3):
        response = await meta_llm.generate_response(prompt=prompt, use_cache=False)
        state = extract_tagged_text(str(response["response"]), "state")
        outputs.append(state)

    tokenized_outputs = [tokenize_words(text) for text in outputs]
    successful = 0

    for index, words in enumerate(tokenized_outputs):
        if not words:
            continue
        similarities = []
        for other_index, other in enumerate(tokenized_outputs):
            if index != other_index:
                similarities.append(jaccard_similarity(words, other))
        max_similarity = max(similarities) if similarities else 0.0
        diverse_enough = max_similarity <= 0.75
        emotion_ok = has_emotion_signal(outputs[index], words, analyzer)
        if diverse_enough and emotion_ok:
            successful += 1

    return successful / len(outputs), outputs


async def test_relevant_field_classification(meta_llm: LLM) -> tuple[float, list[str]]:
    """Evaluate field classification with fractional credit and valid none handling."""
    examples = [
        ("What brought you in today?", []),
        (
            "Can you tell me more about your current symptoms?",
            ["present_illness_positive"],
        ),
        ("Do you smoke or drink?", ["tobacco", "alcohol"]),
    ]

    total_score = 0.0
    responses: list[str] = []
    for question, expected in examples:
        prompt = PWP_CLASS.format(
            available_fields=", ".join(AVAILABLE_FIELDS),
            question=question,
        )

        try:
            response = await meta_llm.generate_response(
                prompt=prompt,
                outlines_class=StringFields,
            )
            fields = parse_fields_response(response["response"])
        except Exception:
            response = await meta_llm.generate_response(prompt=prompt)
            fields = parse_fields_response(response["response"])

        normalized = sorted(set(field for field in fields if field in AVAILABLE_FIELDS))
        responses.append(", ".join(normalized) if normalized else "none")

        expected_set = set(expected)
        predicted_set = set(normalized)

        if not expected_set:
            example_score = 1.0 if not predicted_set else 0.0
        elif not predicted_set:
            example_score = 0.0
        else:
            example_score = len(predicted_set & expected_set) / len(predicted_set)

        total_score += example_score

    return total_score / len(examples), responses


async def test_basic_simulation(
    llm: LLM,
    case_description: dict[str, Any],
) -> tuple[float, list[str]]:
    """Evaluate 3-turn CraftMD-style baseline responses."""
    questions = [
        "What brings you in today?",
        "Tell me more about your medical history.",
        "Do you smoke or drink?",
    ]
    first_person_words = {"i", "i'm", "ive", "i've", "me", "my", "mine"}
    system_instruction = CRAFTMD_SYS.format(
        case_desc=format_case_description(case_description)
    )

    conversation_history: list[dict[str, str]] = []
    successful = 0
    responses: list[str] = []

    for question in questions:
        prompt = conversation_history + [{"role": "user", "content": question}]
        response = await llm.generate_response(
            prompt=prompt,
            system_instruction=system_instruction,
        )
        answer = extract_tagged_text(str(response["response"]), "response")
        responses.append(answer)
        words = tokenize_words(answer)

        has_first_person = any(word in first_person_words for word in words)
        in_length_band = 5 <= len(words) <= 100

        if has_first_person and in_length_band:
            successful += 1

        conversation_history.append({"role": "user", "content": question})
        conversation_history.append({"role": "assistant", "content": answer})

    return successful / len(questions), responses


async def test_personality_steering(
    llm: LLM,
    meta_llm: LLM,
    case_description: dict[str, Any],
) -> tuple[float, dict[str, str]]:
    """Evaluate agreeableness and openness steering under controlled prompts."""
    analyzer = SentimentIntensityAnalyzer()
    cooperative_phrases = ["sure", "okay", "of course", "i can"]
    hostile_phrases = [
        "why",
        "already told you",
        "don't want to",
        "none of your business",
    ]
    tests_passed, total_tests = 0, 5
    question = "Tell me again what brought you in today."
    second_question = "Your symptoms indicate pulmonary embolism."
    prior_belief = "believes he has a muscle strain from moving furniture"

    default_sys = await build_everyday_system_prompt(
        case_description=case_description,
        meta_llm=meta_llm,
        agreeableness_level=1,
        openness_level=1,
    )
    hostile_sys = await build_everyday_system_prompt(
        case_description=case_description,
        meta_llm=meta_llm,
        agreeableness_level=3,
        openness_level=1,
    )
    dogmatic_sys = await build_everyday_system_prompt(
        case_description=case_description,
        meta_llm=meta_llm,
        agreeableness_level=1,
        openness_level=3,
        prior_belief=prior_belief,
    )

    default_resp = await llm.generate_response(
        prompt=[{"role": "user", "content": question}],
        system_instruction=default_sys,
    )
    hostile_resp = await llm.generate_response(
        prompt=[{"role": "user", "content": question}],
        system_instruction=hostile_sys,
    )
    default_text = extract_tagged_text(str(default_resp["response"]), "response")
    hostile_text = extract_tagged_text(str(hostile_resp["response"]), "response")
    responses = {
        "default_turn1": default_text,
        "hostile_turn1": hostile_text,
    }

    cooperative_len = len(tokenize_words(default_text))
    hostile_len = len(tokenize_words(hostile_text))
    cooperative_neg = analyzer.polarity_scores(default_text)["neg"]
    hostile_neg = analyzer.polarity_scores(hostile_text)["neg"]
    cooperative_phrase_hits = sum(
        1 for phrase in cooperative_phrases if phrase in default_text.lower()
    )
    hostile_phrase_hits = sum(
        1 for phrase in hostile_phrases if phrase in hostile_text.lower()
    )

    if cooperative_len > hostile_len:
        tests_passed += 1
    if hostile_neg > cooperative_neg:
        tests_passed += 1
    if cooperative_phrase_hits > hostile_phrase_hits:
        tests_passed += 1

    default_history = [
        {"role": "user", "content": question},
        {"role": "assistant", "content": default_text},
        {"role": "user", "content": second_question},
    ]
    dogmatic_turn1 = await llm.generate_response(
        prompt=[{"role": "user", "content": question}],
        system_instruction=dogmatic_sys,
    )
    dogmatic_answer1 = extract_tagged_text(str(dogmatic_turn1["response"]), "response")
    dogmatic_history = [
        {"role": "user", "content": question},
        {"role": "assistant", "content": dogmatic_answer1},
        {"role": "user", "content": second_question},
    ]

    print(f"Dogmatic system prompt: {dogmatic_sys}")
    print(f"Dogmatic history: {dogmatic_history}")

    default_turn2 = await llm.generate_response(
        prompt=default_history,
        system_instruction=default_sys,
    )
    dogmatic_turn2 = await llm.generate_response(
        prompt=dogmatic_history,
        system_instruction=dogmatic_sys,
    )

    default_answer2 = extract_tagged_text(str(default_turn2["response"]), "response")
    dogmatic_answer2 = extract_tagged_text(str(dogmatic_turn2["response"]), "response")
    responses["dogmatic_turn1"] = dogmatic_answer1
    responses["default_turn2"] = default_answer2
    responses["dogmatic_turn2"] = dogmatic_answer2

    default_neg = analyzer.polarity_scores(default_answer2)["neg"]
    dogmatic_neg = analyzer.polarity_scores(dogmatic_answer2)["neg"]

    prior_tokens = {"muscle", "strain", "moving", "furniture"}
    dogmatic_tokens = set(tokenize_words(dogmatic_answer2))
    prior_belief_present = len(prior_tokens & dogmatic_tokens) >= 2

    if dogmatic_neg > default_neg:
        tests_passed += 1
    if prior_belief_present:
        tests_passed += 1

    return tests_passed / total_tests, responses
