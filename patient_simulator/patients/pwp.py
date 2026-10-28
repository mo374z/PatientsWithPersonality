import asyncio
import hashlib
import json
import os
import random
from typing import Any, Literal, Optional

import pydantic

from patient_simulator.misc.llm import LLM
from patient_simulator.patients.base_patient import BasePatient
from patient_simulator.prompts.patient_prompts import (
    PWP_LATENT_ROLE,
    PWP_SYS,
    PWP_H,
    PWP_H_DOWNPLAY,
    PWP_E,
    PWP_X,
    PWP_A,
    PWP_C,
    PWP_C_FUZZY,
    PWP_O,
    PWP_O_META,
    PWP_E_META,
    PWP_CLASS_FORM,
    PWP_REGISTER,
    PWP_FORM,
)

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
LEISURE_FIELDS = ["tobacco", "alcohol", "illicit_drug", "sexual_history", "exercise"]
MEDICAL_FIELDS = [
    "allergies",
    "family_medical_history",
    "medical_device",
    "medical_history",
    "present_illness_positive",
    "present_illness_negative",
    "pain",
    "medication",
]


PHRASE_SAMPLING = {"temperature": 0.3, "max_tokens": 64, "stop": ["</phrase>"]}

_PHRASE_CACHE: dict[str, str] = {}


async def generate_phrase(llm: LLM, prompt: str, attempts: int) -> str:
    """Rewrite one case value into a patient-facing phrase, memoised on the prompt.

    The rewrite depends only on the field value, so the same value costs one call
    however many personas or prompt candidates are evaluated against it.
    """
    if prompt in _PHRASE_CACHE:
        return _PHRASE_CACHE[prompt]

    for _ in range(attempts):
        response = await llm.generate_response(
            prompt=[{"role": "user", "content": prompt}],
            sampling_kwargs=PHRASE_SAMPLING,
        )
        text = (
            response["response"].split("<phrase>")[-1].replace("</phrase>", "").strip()
        )
        if 1 <= len(text.split()) <= 10:
            break

    _PHRASE_CACHE[prompt] = text
    return text


def downplay_prompt(leisure_info: str) -> str:
    return PWP_H_DOWNPLAY.format(leisure_info=leisure_info)


def fuzzy_prompt(field: str, medical_info: str) -> str:
    return PWP_C_FUZZY.format(medical_info=f"{field}: '{medical_info.strip()}'")


def case_phrase_prompts(case: dict[str, Any]) -> dict[str, int]:
    """Every phrase rewrite a case needs, mapped to its retry budget.

    The rewrites depend only on the case values, never on the HEXACO parametrisation,
    so they can be generated once per case and replayed into the memo cache.
    """
    prompts = {
        downplay_prompt(case[field]): 3
        for field in LEISURE_FIELDS
        if case[field] != "Unknown"
    }
    prompts.update(
        {
            fuzzy_prompt(field, value): 1
            for field in MEDICAL_FIELDS
            if case[field] != "Unknown"
            for value in case[field].split(";")
        }
    )
    return prompts


async def precompute_case_phrases(llm: LLM, case: dict[str, Any]) -> dict[str, str]:
    """Generate a case's phrase rewrites concurrently, as a prompt to phrase mapping."""
    prompts = case_phrase_prompts(case)
    phrases = await asyncio.gather(
        *(
            generate_phrase(llm=llm, prompt=prompt, attempts=attempts)
            for prompt, attempts in prompts.items()
        )
    )
    return dict(zip(prompts, phrases))


def seed_phrase_cache(phrases: dict[str, str]) -> None:
    _PHRASE_CACHE.update(phrases)


class StringFields(pydantic.BaseModel):
    """Schema for a list of strings."""

    fields: list[str]


class QuestionAnalysis(pydantic.BaseModel):
    """Relevant case fields plus the conversational form of the doctor's turn."""

    fields: list[str]
    form: Literal["polar", "open", "statement"]


class PatientsWithPersonality(BasePatient):
    """A realistic Patient Information Recall and Disclosure Simulator based on HEXACO framework."""

    def __init__(
        self,
        case_description: str | dict[str, Any],
        h: int,
        e: int,
        x: int,
        a: int,
        c: int,
        o: int,
        llm: LLM,
        level: str,
        register: str,
        answer_form: float,
        meta_llm: LLM | None = None,
        dynamic_case_description: bool = True,
        verbosity: int = 0,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            case_description=case_description,
            llm=llm,
        )

        self.h = h
        self.e = e
        self.x = x
        self.a = a
        self.c = c
        self.o = o
        self.meta_llm = meta_llm if meta_llm is not None else llm
        self.level = level
        if register not in PWP_REGISTER:
            raise ValueError(
                f"register must be one of {sorted(PWP_REGISTER)}, got {register!r}"
            )
        self.register = register
        if not 0.0 <= float(answer_form) <= 1.0:
            raise ValueError(
                f"answer_form must be a rate in [0, 1], got {answer_form!r}"
            )
        self.answer_form = float(answer_form)
        self._form_rng = random.Random(
            int(
                hashlib.md5(
                    json.dumps(
                        self.case_description, sort_keys=True, default=str
                    ).encode()
                ).hexdigest()[:8],
                16,
            )
        )
        self.last_question_form = "open"
        self.dynamic_case_description = dynamic_case_description
        self.verbosity = verbosity
        self.base_case_description = dict(self.case_description)

        try:
            self.age = int(self.case_description["age"])
        except (ValueError, TypeError, KeyError):
            raise ValueError(
                "Case description must include a valid 'age' field for PatientsWithPersonality."
            )

        self.tangent_topics = self._load_tangent_topics()

        self.tangent_topic = ""
        self.prior_belief = ""
        self.current_emotional_state = ""
        self.last_relevant_fields = []
        self.downplayed_fields = {}
        self.fuzzy_history = {}
        self.latent_role_description = ""
        self._initialized = False

    async def _retry_value_error_generation(self, generator, label: str) -> str:
        """Retry generation on parse-related ValueError up to 3 times."""
        last_error = None
        for attempt in range(1, 4):
            try:
                return await generator()
            except ValueError as exc:
                last_error = exc
                if self.verbosity >= 1:
                    print(
                        f"[RETRY] {label} generation failed (attempt {attempt}/3): {exc}"
                    )

        raise ValueError(
            f"Failed to generate {label} after 3 attempts. Last parse error: {last_error}"
        )

    def _load_tangent_topics(self) -> list:
        tangent_topics_path = os.path.join(
            os.path.dirname(__file__), "../../data/tangent_topics.json"
        )
        with open(tangent_topics_path, "r") as f:
            return json.load(f)

    async def _generate_prior_belief(self) -> str:
        chiefcomplaint = self.base_case_description["chiefcomplaint"]

        response = await self.meta_llm.generate_response(
            prompt=PWP_O_META.format(chiefcomplaint=chiefcomplaint),
        )

        text = response["response"]
        if "<prior_belief>" in text and "</prior_belief>" in text:
            return text.split("<prior_belief>")[1].split("</prior_belief>")[0].strip()
        elif "<prior_belief>" in text:
            return text.split("<prior_belief>")[1].strip()
        raise ValueError(f"Failed to generate prior belief. Response: {text}")

    async def _generate_emotional_state(self) -> str:
        response = await self.meta_llm.generate_response(
            prompt=PWP_E_META.format(
                personal_information=self._build_personal_information_text()
            ),
        )

        text = response["response"]
        if "<state>" in text and "</state>" in text:
            return text.split("<state>")[1].split("</state>")[0].strip()
        elif "<state>" in text:
            return text.split("<state>")[1].strip()
        else:
            raise ValueError(f"Failed to generate emotional state. Response: {text}")

    async def _initialize_dynamic_prompts(self) -> None:
        if self.x == 3:
            topics = []
            for min_age, max_age, age_topics in self.tangent_topics:
                if min_age <= self.age <= max_age:
                    topics.extend(age_topics)
            if topics:
                self.tangent_topic = random.choice(topics)
                if self.verbosity >= 1:
                    print(f"[INIT] Sampled tangent topic: {self.tangent_topic}")

        if self.o == 3:
            self.prior_belief = await self._retry_value_error_generation(
                self._generate_prior_belief, "prior belief"
            )
            if self.verbosity >= 1:
                print(f"[INIT] Generated prior belief: {self.prior_belief}")

        if self.e == 3:
            self.current_emotional_state = await self._retry_value_error_generation(
                self._generate_emotional_state, "emotional state"
            )
            if self.verbosity >= 1:
                print(
                    f"[INIT] Generated emotional state: {self.current_emotional_state}"
                )

    def _build_personal_information_text(self) -> str:
        lines = []
        for field, value in self.base_case_description.items():
            if field in PERSONAL_FIELDS and not (
                isinstance(value, str) and value.strip().lower() == "unknown"
            ):
                lines.append(f"\t{field}: {value}")

        lines.append(f"\tcefr_level: {self.level}")
        if self.tangent_topic:
            lines.append(f"\tfurther information: {self.tangent_topic}")

        return "\n".join(lines)

    def _build_hexaco_personality_text(self) -> str:
        honesty = PWP_H[self.h]
        emotional_state = self.current_emotional_state if self.e == 3 else PWP_E[self.e]
        extraversion = PWP_X[self.x]
        agreeableness = PWP_A[self.a]
        conscientiousness = PWP_C[self.c]
        openness = (
            PWP_O[3].format(prior_belief=self.prior_belief)
            if self.o == 3 and self.prior_belief
            else PWP_O[self.o]
        )

        return "\n".join(
            [
                f"\tHonesty-Humility: {honesty}",
                f"\tEmotionality: {emotional_state}",
                f"\tExtraversion: {extraversion}",
                f"\tAgreeableness: {agreeableness}",
                f"\tConscientiousness: {conscientiousness}",
                f"\tOpenness: {openness}",
            ]
        )

    def build_role_inputs(self) -> str:
        """The personal and HEXACO text a latent role is generated from."""
        return "\n".join(
            [
                "Personal Information:",
                self._build_personal_information_text(),
                "",
                "HEXACO Personality:",
                self._build_hexaco_personality_text(),
            ]
        )

    async def _generate_latent_role_description(self) -> str:
        prompt = PWP_LATENT_ROLE.format(
            personal_information=self._build_personal_information_text(),
            hexaco_personality=self._build_hexaco_personality_text(),
        )

        response = await self.meta_llm.generate_response(prompt=prompt)
        text = response["response"]

        if "<role>" in text and "</role>" in text:
            role = text.split("<role>")[1].split("</role>")[0].strip()
            if role:
                return role

        raise ValueError(
            f"Failed to generate latent role description. Response: {text}"
        )

    async def _generate_downplayed_fields(self, fields: list[str]) -> dict[str, str]:
        downplayed_fields = {}

        for field in fields:
            if self.case_description[field] == "Unknown":
                downplayed_fields[field] = "Unknown"
            else:
                downplayed_fields[field] = await generate_phrase(
                    llm=self.meta_llm,
                    prompt=downplay_prompt(self.case_description[field]),
                    attempts=3,
                )

        if self.verbosity >= 1:
            print(f"[INIT] Generated downplayed leisure fields: {downplayed_fields}")

        return downplayed_fields

    async def _build_leisure_info_grid(self):
        grid = []
        mask = []

        self.downplayed_fields = await self._generate_downplayed_fields(
            fields=LEISURE_FIELDS
        )

        denied_fields = {
            "tobacco": "non-smoker",
            "alcohol": "not drinking alcohol",
            "illicit_drug": "no drug usage",
            "sexual_history": "no specificites",
            "exercise": "no exercise routine",
        }

        for field in LEISURE_FIELDS:
            grid.append(
                [
                    self.case_description[field],
                    self.downplayed_fields[field],
                    denied_fields[field],
                ]
            )
            if self.h == 1:
                mask.append(0)
            elif self.h == 2:
                mask.append(1 if random.random() < 0.5 else 0)
            elif self.h in [3]:
                mask.append(2 if random.random() < 0.5 else 1)

        self.leisure_info_mask = mask
        self.leisure_info_grid = grid

    async def _generate_fuzzy_history(self, fields: list[str]) -> dict[str, list[str]]:
        fuzzy_med_info = {}

        for field in fields:
            if self.case_description[field] == "Unknown":
                fuzzy_med_info[field] = ["Unknown"]
            else:
                fuzzy_med_info[field] = [
                    await generate_phrase(
                        llm=self.meta_llm,
                        prompt=fuzzy_prompt(field, value),
                        attempts=1,
                    )
                    for value in self.case_description[field].split(";")
                ]

        if self.verbosity >= 1:
            print(f"[INIT] Generated fuzzy medical history: {fuzzy_med_info}")

        return fuzzy_med_info

    async def _build_medical_info_grid(self):
        grid = []
        mask = []

        self.fuzzy_history = await self._generate_fuzzy_history(fields=MEDICAL_FIELDS)

        for field in MEDICAL_FIELDS:
            original = self.case_description[field]
            fuzzy = "; ".join(self.fuzzy_history[field])
            grid.append([original, fuzzy, ""])
            mask.append(self.c)

        self.medical_info_mask = mask
        self.medical_info_grid = grid

    def _update_case_description(self, relevant_fields: Optional[list[str]]):
        if relevant_fields:
            for field in relevant_fields:
                if field in MEDICAL_FIELDS:
                    self.medical_info_mask[MEDICAL_FIELDS.index(field)] -= (
                        1
                        if self.medical_info_mask[MEDICAL_FIELDS.index(field)] > 0
                        else 0
                    )
            for field in relevant_fields:
                if field in LEISURE_FIELDS:
                    index = LEISURE_FIELDS.index(field)
                    self.case_description[field] = self.leisure_info_grid[index][
                        self.leisure_info_mask[index]
                    ]
                elif field in MEDICAL_FIELDS:
                    index = MEDICAL_FIELDS.index(field)
                    self.case_description[field] = self.medical_info_grid[index][
                        self.medical_info_mask[index]
                    ]

    def _build_style_directives(self) -> str:
        directives = [PWP_REGISTER[self.register]]
        if self._form_rng.random() < self.answer_form:
            directives.append(PWP_FORM[self.last_question_form])
        return "".join(d for d in directives if d)

    def _build_system_prompt(self) -> str:
        return PWP_SYS.format(
            latent_role=self.latent_role_description,
            structured_information="\n".join(
                [
                    f"    {field}: {self.case_description[field]}"
                    for field in self.case_description
                ]
            ),
            style_directives=self._build_style_directives(),
        )

    async def _classify_relevant_fields(self, question: str) -> list[str]:
        if isinstance(self.base_case_description, dict):
            available_fields = [
                field
                for field in LEISURE_FIELDS + MEDICAL_FIELDS
                if field in self.base_case_description
            ]
        else:
            return []

        prompt = PWP_CLASS_FORM.format(
            available_fields=", ".join(available_fields), question=question
        )

        response = await self.meta_llm.generate_response(
            prompt=prompt,
            outlines_class=QuestionAnalysis,
        )
        structured_response = response["response"]
        if not isinstance(structured_response, QuestionAnalysis):
            raise TypeError(
                "Expected QuestionAnalysis from structured response, got "
                f"{type(structured_response).__name__} with content: {structured_response}"
            )
        response = structured_response

        self.last_question_form = response.form
        relevant_fields = response.fields

        valid_fields = [field for field in relevant_fields if field in available_fields]

        if self.verbosity >= 2:
            print(
                f"[CLASSIFICATION] Model classified fields: {relevant_fields}. Differing fields: {set(relevant_fields) - set(valid_fields)}"
            )

        if self.verbosity >= 1:
            print(f"[CLASSIFICATION] Relevant fields: {valid_fields}")

        if self.dynamic_case_description:
            self._update_case_description(valid_fields)

        self.last_relevant_fields = valid_fields

        return valid_fields

    async def _ensure_initialized(self) -> None:
        """Lazily initialize expensive fields on first use."""
        if self._initialized:
            return

        if self.verbosity >= 1:
            print("[INIT] Performing lazy initialization...")

        if self.dynamic_case_description:
            await self._build_leisure_info_grid()
            await self._build_medical_info_grid()

        await self._initialize_dynamic_prompts()
        self.latent_role_description = await self._retry_value_error_generation(
            self._generate_latent_role_description, "latent role description"
        )

        if self.dynamic_case_description:
            self.case_description = {}
            self._update_case_description([])
        else:
            self.case_description = {
                field: self.base_case_description[field]
                for field in LEISURE_FIELDS + MEDICAL_FIELDS
            }

        if self.verbosity >= 1:
            print(f"[INIT] Generated latent role: {self.latent_role_description}")

        self._initialized = True

    async def get_response(self, question: str) -> str:
        await self._ensure_initialized()

        await self._classify_relevant_fields(question)

        system_prompt = self._build_system_prompt()

        if self.verbosity >= 2:
            print(f"\n[SYSTEM PROMPT]\n{system_prompt}\n")

        self.conversation_history.append({"role": "user", "content": question})

        response = await self.llm.generate_response(
            prompt=self.conversation_history,
            system_instruction=system_prompt,
            sampling_kwargs={
                "stop": ["</response>", "Question:"],
            },
        )

        response_trimmed = (
            response["response"]
            .split("<response>")[-1]
            .replace("</response>", "")
            .strip()
        )

        self.conversation_history.append(
            {"role": "assistant", "content": response_trimmed}
        )
        return response_trimmed

    def __name__(self, short: bool = True) -> str:
        if short:
            return "PatientsWithPersonality"

        static = "" if self.dynamic_case_description else "Static_"
        return (
            f"PatientsWithPersonality_{static}H{self.h}_E{self.e}_X{self.x}"
            f"_A{self.a}_C{self.c}_O{self.o}_L{self.level}"
        )
