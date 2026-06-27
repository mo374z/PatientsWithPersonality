"""Extraction of a case description from a conversational text file."""

import json
from typing import Any

import pydantic
from patient_simulator.misc.llm import LLM
from patient_simulator.misc.utils import parse_transcript_file


class PatientProfile(pydantic.BaseModel):
    """Structured patient profile for PatientSim."""

    age: str = pydantic.Field(description="Patient's age")
    gender: str = pydantic.Field(description="Patient's gender")
    race: str = pydantic.Field(
        default="Unknown", description="Patient's race/ethnicity"
    )

    tobacco: str = pydantic.Field(default="Unknown", description="Tobacco use history")
    alcohol: str = pydantic.Field(
        default="Unknown", description="Alcohol consumption history"
    )
    illicit_drug: str = pydantic.Field(
        default="Unknown", description="Illicit drug use"
    )
    sexual_history: str = pydantic.Field(
        default="Unknown", description="Sexual history"
    )
    exercise: str = pydantic.Field(default="Unknown", description="Exercise habits")
    marital_status: str = pydantic.Field(
        default="Unknown", description="Marital status"
    )
    children: str = pydantic.Field(
        default="Unknown", description="Information about children"
    )
    living_situation: str = pydantic.Field(
        default="Unknown", description="Living situation"
    )
    occupation: str = pydantic.Field(default="Unknown", description="Occupation")
    insurance: str = pydantic.Field(default="Unknown", description="Insurance coverage")

    allergies: str = pydantic.Field(
        default="No known allergies", description="Known allergies"
    )
    family_medical_history: str = pydantic.Field(
        default="Unknown", description="Family medical history"
    )
    medical_device: str = pydantic.Field(
        default="None", description="Medical devices used"
    )
    medical_history: str = pydantic.Field(
        default="Unknown", description="Past medical history"
    )

    present_illness_positive: str = pydantic.Field(
        description="Positive findings for current illness"
    )
    present_illness_negative: str = pydantic.Field(
        default="", description="Denied or negative findings"
    )
    chiefcomplaint: str = pydantic.Field(description="Chief complaint")
    pain: str = pydantic.Field(default="0", description="Pain level 0-10")
    medication: str = pydantic.Field(default="None", description="Current medications")
    arrival_transport: str = pydantic.Field(
        default="Unknown", description="How patient arrived"
    )
    disposition: str = pydantic.Field(default="Unknown", description="Disposition")
    diagnosis: str = pydantic.Field(default="Unknown", description="Diagnosis")


CASE_DESCRIPTION_EXTRACTION_PROMPT = """You are a specialized medical scribe assistant. Your task is to extract a structured, clinically precise patient profile from the following clinical case description.

### Guidelines for Extraction:

1.  **Strict Adherence to Facts:** Extract ONLY information explicitly stated in the case description. Do not infer diagnoses, assume unstated details, or fill gaps with general knowledge. If a specific detail is not mentioned, return "Unknown".
2.  **Neutral & Objective Tone:** Use a detached, clinical tone. Avoid emotive language, subjective interpretations, or judgment.
3.  **Telegraphic Style:**
    * Eliminate pronouns (I, he, she), articles (a, an, the), and unnecessary verbs.
    * Use noun phrases and standard medical terminology where applicable (e.g., replace "trouble breathing" with "dyspnea" only if clinically certain; otherwise use exact descriptive phrasing like "shortness of breath").
4.  **Granularity & Detail:**
    * Capture specific modifiers: severity, duration, frequency, dosage, location, and radiation.
    * *Example:* Instead of "Headache", use "pulsating frontal headache; 3-day duration; 7/10 severity".
5.  **Formatting:**
    * Separate distinct items within a single field using semicolons (e.g., "condition A; condition B").
    * Output strictly valid JSON.

### Clinical Case Description:
{case_description}

### Target Information Structure:

Demographics:
- age: Patient's age in years
- gender: Patient's gender
- race: Patient's race/ethnicity (if mentioned)

Social History:
- tobacco: Tobacco/smoking history (e.g., "Smokes 1 pack per day for 10 years", "Non-smoker")
- alcohol: Alcohol consumption (e.g., "10 drinks per week", "Occasional drinker", "Non-drinker")
- illicit_drug: Recreational drug use (e.g., "Cannabis 5mg per week", "None")
- sexual_history: Sexual history if discussed
- exercise: Exercise habits (e.g., "Runs 30 minutes every other day")
- marital_status: Marital status
- children: Information about children
- living_situation: Living situation/conditions
- occupation: Job/occupation
- insurance: Insurance information

Medical History:
- allergies: Known allergies or "No known allergies"
- family_medical_history: Family medical conditions (e.g., "Father had heart attack at 45, father had high cholesterol")
- medical_device: Medical devices used (e.g., "None", "Insulin pump")
- medical_history: Past medical conditions, surgeries, chronic illnesses (e.g., "Type 2 diabetes for 5 years")

Current Visit:
- present_illness_positive: All positive symptoms and findings mentioned (describe in detail)
- present_illness_negative: All denied or negative symptoms (e.g., "No radiation of pain, no shortness of breath")
- chiefcomplaint: Main reason for visit
- pain: Pain level 0-10 if mentioned, otherwise "0"
- medication: Current medications being taken
- arrival_transport: How patient arrived (ambulance, walked in, etc.)
- disposition: Expected outcome or plan
- diagnosis: Working or final diagnosis if stated

**Output:** Provide the extracted information in JSON format."""

EXTRACTION_PROMPT = """You are a specialized medical scribe assistant. Your task is to extract a structured, clinically precise summary from the following doctor-patient conversation transcript.

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
    * Separate distinct items within a single field using semicolons (e.g., "condition A; condition B").
    * Output strictly valid JSON.

### Conversation Transcript:
{transcript}

### Target Information Structure:

Demographics:
- age: Patient's age in years
- gender: Patient's gender
- race: Patient's race/ethnicity (if mentioned)

Social History:
- tobacco: Tobacco/smoking history (e.g., "Smokes 1 pack per day for 10 years", "Non-smoker")
- alcohol: Alcohol consumption (e.g., "10 drinks per week", "Occasional drinker", "Non-drinker")
- illicit_drug: Recreational drug use (e.g., "Cannabis 5mg per week", "None")
- sexual_history: Sexual history if discussed
- exercise: Exercise habits (e.g., "Runs 30 minutes every other day")
- marital_status: Marital status
- children: Information about children
- living_situation: Living situation/conditions
- occupation: Job/occupation
- insurance: Insurance information

Medical History:
- allergies: Known allergies or "No known allergies"
- family_medical_history: Family medical conditions (e.g., "Father had heart attack at 45, father had high cholesterol")
- medical_device: Medical devices used (e.g., "None", "Insulin pump")
- medical_history: Past medical conditions, surgeries, chronic illnesses (e.g., "Type 2 diabetes for 5 years")

Current Visit:
- present_illness_positive: All positive symptoms and findings mentioned (describe in detail)
- present_illness_negative: All denied or negative symptoms (e.g., "No radiation of pain, no shortness of breath")
- chiefcomplaint: Main reason for visit
- pain: Pain level 0-10 if mentioned, otherwise "0"
- medication: Current medications being taken
- arrival_transport: How patient arrived (ambulance, walked in, etc.)
- disposition: Expected outcome or plan
- diagnosis: Working or final diagnosis if stated

**Output:** Provide the extracted information in JSON format."""


def _parse_patient_profile_response(
    raw_response: Any,
    patient_profile_class: type[pydantic.BaseModel],
) -> pydantic.BaseModel:
    if isinstance(raw_response, patient_profile_class):
        return raw_response

    if isinstance(raw_response, pydantic.BaseModel):
        return patient_profile_class.model_validate(raw_response.model_dump())

    if isinstance(raw_response, dict):
        return patient_profile_class.model_validate(raw_response)

    if isinstance(raw_response, str):
        try:
            return patient_profile_class.model_validate_json(raw_response)
        except pydantic.ValidationError:
            parsed = json.loads(raw_response)
            return patient_profile_class.model_validate(parsed)

    raise TypeError(
        "Unsupported response type for profile extraction: "
        f"{type(raw_response).__name__}"
    )


async def extract_patient_profile_from_transcript(
    transcript_path: str,
    llm: LLM,
    extraction_prompt: str = EXTRACTION_PROMPT,
    patient_profile_class: type[pydantic.BaseModel] = PatientProfile,
) -> dict:
    """Extract structured patient profile from a conversation transcript file."""
    turns = parse_transcript_file(transcript_path)

    if not turns:
        raise ValueError(f"No valid turns found in transcript: {transcript_path}")

    transcript_text = "\n".join([f"{speaker}: {text}" for speaker, text in turns])

    prompt = extraction_prompt.format(transcript=transcript_text)

    response = await llm.generate_response(
        prompt=prompt,
        outlines_class=patient_profile_class,
    )

    profile = _parse_patient_profile_response(
        response["response"],
        patient_profile_class,
    )

    return profile.model_dump()


async def extract_patient_profile_from_text(
    transcript_text: str,
    llm: LLM,
    extraction_prompt: str = EXTRACTION_PROMPT,
    patient_profile_class: type[pydantic.BaseModel] = PatientProfile,
) -> dict:
    """Extract structured patient profile from conversation text."""
    prompt = extraction_prompt.format(transcript=transcript_text)

    response = await llm.generate_response(
        prompt=prompt,
        outlines_class=patient_profile_class,
    )

    profile = _parse_patient_profile_response(
        response["response"],
        patient_profile_class,
    )

    return profile.model_dump()


async def extract_patient_profile_from_case_description(
    case_description: str,
    llm: LLM,
    extraction_prompt: str = CASE_DESCRIPTION_EXTRACTION_PROMPT,
    patient_profile_class: type[pydantic.BaseModel] = PatientProfile,
) -> dict:
    """Extract structured patient profile from a clinical case description."""
    prompt = extraction_prompt.format(case_description=case_description)

    response = await llm.generate_response(
        prompt=prompt,
        outlines_class=patient_profile_class,
    )

    profile = _parse_patient_profile_response(
        response["response"],
        patient_profile_class,
    )

    return profile.model_dump()
