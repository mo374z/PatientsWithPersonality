import pydantic


class CaseDescription(pydantic.BaseModel):
    """Structured case description."""

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
    allergies: str = pydantic.Field(default="Unknown", description="Known allergies")
    family_medical_history: str = pydantic.Field(
        default="Unknown", description="Family medical history"
    )
    medical_device: str = pydantic.Field(
        default="Unknown", description="Medical devices used"
    )
    medical_history: str = pydantic.Field(
        default="Unknown", description="Past medical history"
    )
    present_illness_positive: str = pydantic.Field(
        default="Unknown", description="Positive findings for current illness"
    )
    present_illness_negative: str = pydantic.Field(
        default="Unknown", description="Denied or negative findings"
    )
    pain: str = pydantic.Field(default="Unknown", description="Pain level")
    medication: str = pydantic.Field(
        default="Unknown", description="Current medications"
    )


EVAL_CASE_EXTRACTION_PROMPT = """You are a specialized medical scribe assistant. Your task is to extract a structured, clinically precise summary from the following doctor-patient conversation transcript.

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

Social Fields:
- tobacco: Tobacco/smoking history (e.g., "Smokes 1 pack per day for 10 years", "Non-smoker")
- alcohol: Alcohol consumption (e.g., "10 drinks per week", "Occasional drinker", "Non-drinker")
- illicit_drug: Recreational drug use (e.g., "Cannabis 5mg per week", "None")
- sexual_history: Sexual history if discussed
- exercise: Exercise habits (e.g., "Runs 30 minutes every other day")

Medical Fields:
- allergies: Known allergies or "No known allergies"
- family_medical_history: Family medical conditions (e.g., "Father had heart attack at 45, father had high cholesterol")
- medical_device: Medical devices used (e.g., "None", "Insulin pump")
- medical_history: Past medical conditions, surgeries, chronic illnesses (e.g., "Type 2 diabetes for 5 years")
- present_illness_positive: All positive symptoms and findings mentioned (describe in detail)
- present_illness_negative: All denied or negative symptoms (e.g., "No radiation of pain, no shortness of breath")
- pain: Pain level 0-10
- medication: Current medications being taken

**Output:** Provide the extracted information in JSON format."""

# Profile Fidelity Evaluation
PROFILE_FIDELITY_PROMPT = """You are evaluating whether a simulated patient's profile information matches the real patient information.

Categories:
- 'Supported': The simulated information is present in the real patient information.
- 'Implied': The simulated information is not explicitly stated but can be reasonably inferred from the real patient information.
- 'Contradiction': The simulated information directly contradicts the real patient information.
- 'Hallucination': The simulated information is not present in the real patient information and cannot be reasonably inferred.

Real Patient Information: {real_item}
Simulated Patient Information: {simulated_item}

Provide your classification between the tags <category> and </category>. Do not include any other text in your answer."""


FACT_FIDELITY_PROMPT = """You are checking whether each patient-stated fact is supported by the provided case description.

Task:
- Evaluate every fact in the given list.
- Keep exact list order.
- For each list position, return true if the fact is supported by or directly entailed by the case description.
- Return false if the fact is not supported.

Case description fields:
{personal_fields}
{case_fields}

Patient facts:
{facts}

Return a structured output with one boolean per fact position. The output list length must exactly match the number of input facts. Do not include explanations."""


PERSONAL_INFO_PRESENCE_PROMPT = """You are checking whether each personal information field is represented in a latent patient role description.

Task:
- Evaluate every personal information field in the given list.
- Keep exact list order.
- For each list position, return true if the field value is reflected in the latent role description.
- Return false if the field value is missing, contradicted, or not inferable from the latent role description.
- Use semantic matching, not exact string matching. Paraphrases and equivalent wording should count as true.

Personal information fields:
{personal_info_fields}

Latent role description:
{latent_role}

Return a structured output with one boolean per field position. The output list length must exactly match the number of input fields. Do not include explanations."""


# Response Quality Evaluation - Relevance
RELEVANCE_EVALUATION_PROMPT = """You are evaluating a simulated patient's response to a doctor's question. Classify the patient's response based on its relevance to the doctor's clinical inquiry.

Categories:
- 'Direct Answer': Directly addresses the clinical inquiry.
- 'Tangential': Addresses the topic but adds unnecessary fluff or misses the core question.
- 'Irrelevant/Evasive': Completely ignores the medical inquiry or hallucinates a new topic.

Doctor's Question: {doctor_question}
Patient's Response: {patient_response}

Provide your classification between the tags <category> and </category>. Do not include any other text in your answer."""


# Response Quality Evaluation - Realism (Content)
REALISM_CONTENT_EVALUATION_PROMPT = """You are evaluating how realistic a simulated patient response is compared to a real patient response. Focus ONLY on clinical content (symptoms, negations, qualifiers). Ignore style, length, and grammar.

Categories:
- 'Similar': Clinical content closely matches the real response.
- 'Dissimilar': Clinical content differs significantly from the real response.

Real Patient Response: {real_response}
Simulated Patient Response: {patient_response}

Provide your classification between the tags <category> and </category>. Do not include any other text in your answer."""

# Response Quality Evaluation - Realism (Style)
REALISM_STYLE_EVALUATION_PROMPT = """You are evaluating how realistic a simulated patient response is compared to a real patient response. Focus ONLY on style (brevity, tone, formality, sentence length). Ignore medical facts and clinical content.

Categories:
- 'Similar': Style matches the real response.
- 'Dissimilar': Style differs significantly from the real response.

Real Patient Response: {real_response}
Simulated Patient Response: {patient_response}

Provide your classification between the tags <category> and </category>. Do not include any other text in your answer."""

# Persona Consistency Evaluation
PERSONA_CONSISTENCY_EVALUATION_PROMPT = """You are evaluating whether a simulated patient maintains a consistent personality and communication style throughout a conversation.

Categories:
- 'Consistent': The patient's personality traits, tone, and style remain stable across responses.
- 'Inconsistent': The patient's personality traits, tone, or style noticeably shift between responses.

First patient responses: {first_responses}
Last patient responses: {last_responses}

Provide your classification between the tags <category> and </category>. Do not include any other text in your answer."""

PERSONALITY_RECON_H_PROMPT = """Honesty / Disclosure (H) axis levels:
- '1': Transparent and open; freely discloses sensitive habits and details.
- '2': Mostly truthful but hesitant; downplays or delays disclosure.
- '3': Conceals or distorts key information; evasive or contradictory.

"""

PERSONALITY_RECON_E_PROMPT = """Emotional State (E) axis levels:
- '1': Calm and emotionally steady.
- '2': Anxious and reassurance-seeking.
- '3': Marked distress pattern (catastrophizing, irritability, avoidance, or strong dysregulation).

"""

PERSONALITY_RECON_X_PROMPT = """Extraversion (X) axis levels:
- '1': Minimal, terse responses.
- '2': Concise but complete on-topic responses.
- '3': Talkative pattern with anecdotes, tangents, or frequent back-questions.

"""

PERSONALITY_RECON_A_PROMPT = """Agreeableness (A) axis levels:
- '1': Cooperative and trusting.
- '2': Guarded and cautious.
- '3': Frustrated, confrontational, or strongly skeptical.

"""

PERSONALITY_RECON_C_PROMPT = """Conscientiousness (C) axis levels:
- '1': Precise recall of dates, timelines, doses, and details.
- '2': Approximate recall with some fuzziness.
- '3': Disorganized recall with frequent uncertainty or inconsistency.

"""

PERSONALITY_RECON_O_PROMPT = """Openness / Prior Beliefs (O) axis levels:
- '1': Open to medical explanations and updates beliefs.
- '2': Mild skepticism; prefers familiar/home-remedy framing.
- '3': Dogmatic prior belief or self-diagnosis; dismisses alternatives.

"""


TAGGED_FACT_EXTRACTION_PROMPT = """You are a specialized medical scribe. Extract every health-related atomic fact stated by the patient in a single turn, tag each fact with the case description field it belongs to, and judge whether each fact is supported by the case description.

Guidelines:
1. Extract ONLY information explicitly stated by the patient in the given turn. Do not infer.
2. Each fact must be atomic (one checkable statement).
3. For each fact, set `field` to one of the available fields listed below that best matches the fact. If no listed field applies, use "other".
4. For each fact, set `supported` to true if the fact is supported by or directly entailed by the case description value for its field; false otherwise. Facts tagged "other" should be judged against the full case description and set to false if unsupported.
5. Use telegraphic style; keep specific modifiers (severity, duration, frequency, dosage).
6. Return a structured list. Preserve the order in which facts appear in the patient turn.

Available fields: {available_fields}

Case description (ground truth):
{case_fields}

Patient turn:
{patient_turn}
"""


REALISM_JUDGE_PROMPT = """You are a physician reviewer assessing how convincingly this medical consultation reads as a real patient encounter. Rate the patient's communication on the dimensions below. Use the full 1–5 range — a 3 is the baseline for a patient who is mostly convincing with minor awkward moments.

Dimensions:

1. p_real (0.0–1.0): Your overall probability that this is a real patient. 0.5 means you cannot tell. Do not anchor to 0.0 or 1.0 unless the evidence is overwhelming.

2. symptom_realism (1–5): Do the symptoms fit the clinical context and emerge naturally?
   1 = clinically implausible or strangely volunteered; 3 = plausible, minor phrasing oddities; 5 = reported the way real patients naturally describe symptoms.

3. information_control (1–5): Does the patient's level of medical detail feel appropriate for a layperson?
   Patients vary — some look things up and use clinical terms, others stay vague. Only penalize when the level of expertise is implausibly systematic (e.g. reciting diagnostic criteria, naming differentials unprompted).
   1 = medical knowledge is implausibly precise and comprehensive; 3 = some clinical language, within the range of an informed patient; 5 = feels natural for someone describing their health.

4. style_realism (1–5): Does the conversational style, language, and phrasing sound like a real person?
   1 = reads like a structured report; 3 = natural with occasional stiff phrasing; 5 = spontaneous, human-sounding dialogue.

5. justification: One short paragraph explaining your overall impression and any specific turns that shaped your rating.

Conversation:
{transcript}

Return the structured output with all fields populated.
"""
