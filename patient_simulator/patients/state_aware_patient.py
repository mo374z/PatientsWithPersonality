from typing import Any

from patient_simulator.misc.llm import LLM
from patient_simulator.patients.base_patient import (
    BasePatient,
    _normalize_case_description,
)
from patient_simulator.prompts.patient_prompts import (
    SAPS_CATEGORY,
    SAPS_INQUIRY_SPECIFICITY,
    SAPS_ADVICE_SPECIFICITY,
    SAPS_INQUIRY_RELEVANCE,
    SAPS_ADVICE_RELEVANCE,
    SAPS_WORKING_MEMORY_REQUIREMENTS,
)


class StateAwarePatient(BasePatient):
    """State Aware Patient Simulator (SAPS) with state tracking and memory bank (Liao et al., 2024)

    Overview of the state codes:
    - A-A-A: Effective Inquiry (specific question with relevant info)
    - A-A-B: Ineffective Inquiry (specific but no relevant info)
    - A-B: Ambiguous Inquiry (too broad)
    - B-A-A: Effective Advice (with relevant results)
    - B-A-B: Ineffective Advice (without relevant results)
    - B-B: Ambiguous Advice (too broad)
    - C: Demand (physical action)
    - D: Other Topics (non-medical)
    - E: Conclusion (end of consultation)

    """

    def __init__(
        self,
        case_description: str | dict[str, Any],
        llm: LLM,
    ):
        self.llm = llm
        self.case_description = _normalize_case_description(case_description)
        self.conversation_history = []  # Standard conversation history format

        # Memory bank components
        self.long_term_memory = self.case_description  # Patient information (Mlong)
        self.working_memory = (
            SAPS_WORKING_MEMORY_REQUIREMENTS  # Requirements for each state (Mwork)
        )
        self.short_term_memory = []  # Conversation history (Mshort)

        self.is_first_turn = True
        self.last_state_code = None
        self.current_instruction = None

    async def _classify_state_category(self, question: str) -> str:
        """
        Classify the doctor's question into one of five main categories.
        """
        prompt = SAPS_CATEGORY.format(question=question)

        response = await self.llm.generate_response(
            prompt=prompt,
            system_instruction=None,
        )

        # Extract the category letter (A, B, C, D, or E)
        category = response["response"].strip()
        for letter in ["A", "B", "C", "D", "E"]:
            if letter in category:
                return letter

        # Default to inquiry if unclear
        return "A"

    async def _classify_specificity(self, question: str, category: str) -> str:
        """
        Classify if an inquiry or advice is specific or ambiguous.
        Returns: 'specific' or 'ambiguous'
        """
        if category == "A":  # Inquiry
            prompt = SAPS_INQUIRY_SPECIFICITY.format(question=question)
        elif category == "B":  # Advice
            prompt = SAPS_ADVICE_SPECIFICITY.format(question=question)
        else:
            return "specific"  # Not applicable for other categories

        response = await self.llm.generate_response(
            prompt=prompt,
            system_instruction=None,
        )

        result = response["response"].strip()

        if (
            "[Ambiguous]" in result
            or "[Broad]" in result
            or "Ambiguous" in result
            or "Broad" in result
        ):
            return "ambiguous"
        else:
            return "specific"

    async def _check_relevance(self, question: str, category: str) -> str:
        """
        Check if patient information contains relevant information to answer the question.
        Returns: relevant information text or '[No Relevant Information]'
        """
        if category == "A":  # Inquiry
            prompt = SAPS_INQUIRY_RELEVANCE.format(
                patient_info=self.long_term_memory, question=question
            )
        elif category == "B":  # Advice
            prompt = SAPS_ADVICE_RELEVANCE.format(
                patient_info=self.long_term_memory, question=question
            )
        else:
            return "[No Relevant Information]"

        response = await self.llm.generate_response(
            prompt=prompt,
            system_instruction=None,
        )

        return response["response"].strip()

    async def _track_state(self, question: str) -> tuple[str, str]:
        """
        Track the current state of the conversation.
        Returns: (state_code, extracted_memory)
        State codes: A-A-A, A-A-B, A-B, B-A-A, B-A-B, B-B, C, D
        """
        category = await self._classify_state_category(question)

        if category == "A":  # Inquiry
            specificity = await self._classify_specificity(question, category)

            if specificity == "ambiguous":
                return "A-B", ""
            else:
                relevant_info = await self._check_relevance(question, category)

                if "[No Relevant Information]" in relevant_info:
                    return "A-A-B", ""
                else:
                    return "A-A-A", relevant_info

        elif category == "B":  # Advice
            specificity = await self._classify_specificity(question, category)

            if specificity == "ambiguous":
                return "B-B", ""
            else:
                relevant_info = await self._check_relevance(question, category)

                if "[No Relevant Information]" in relevant_info:
                    return "B-A-B", ""
                else:
                    return "B-A-A", relevant_info

        elif category == "C":  # Demand
            return "C", ""

        elif category == "D":  # Other Topics
            return "D", ""

        elif category == "E":  # Conclusion
            return "E", ""

        # Default fallback
        return "A-A-A", self.long_term_memory

    def _format_short_term_memory(self) -> str:
        """Format conversation history for the response generator"""
        if not self.short_term_memory:
            return ""

        history_str = ""
        for exchange in self.short_term_memory:
            history_str += f"Doctor: {exchange['doctor']}\n"
            history_str += f"Patient: {exchange['patient']}\n"

        return history_str

    async def _generate_response(
        self, question: str, state_code: str, extracted_memory: str
    ) -> str:
        """Generate patient response using the response generator"""

        working_template = self.working_memory[state_code]

        patient_info = extracted_memory if extracted_memory else self.long_term_memory
        working_requirement = working_template.format(patient_info=patient_info)
        self.current_instruction = working_requirement

        short_term_str = self._format_short_term_memory()

        if short_term_str:
            prompt = working_requirement + short_term_str + f"Doctor: {question}\n"
        else:
            prompt = working_requirement + f"Doctor: {question}\n"

        response = await self.llm.generate_response(
            prompt=prompt
            + "Provide your answer to the doctor's inquiry between the tags <response> and </response>. Do not include any other text in your answer.",
            system_instruction=None,
            sampling_kwargs={
                "stop": ["</response>", "Doctor:"],
            },
        )

        response_trimmed = (
            response["response"]
            .split("<response>")[-1]
            .replace("</response>", "")
            .strip()
        )

        return response_trimmed

    async def get_response(self, question: str) -> str:
        """
        Get patient response to doctor's question using state-aware approach.
        Main interface matching other patient simulator classes.
        """
        if self.is_first_turn:
            prompt = f"<Patient Condition>: {self.long_term_memory}\n<Response Requirement>: Briefly describe your main symptoms and primary concerns.\nBelow is a dialogue between a doctor and a patient. The patient will respond to the doctor's question in the first person.\nDoctor: {question}\n"

            response = await self.llm.generate_response(
                prompt=prompt
                + "Provide your answer to the doctor's inquiry between the tags <response> and </response>. Do not include any other text in your answer.",
                system_instruction=None,
                sampling_kwargs={
                    "stop": ["</response>", "Doctor:"],
                },
            )

            response_trimmed = (
                response["response"]
                .split("<response>")[-1]
                .replace("</response>", "")
                .strip()
            )

            # Update both short_term_memory (for state tracking) and conversation_history (for parent class)
            self.short_term_memory.append(
                {"doctor": question, "patient": response_trimmed}
            )
            self.conversation_history.append({"role": "user", "content": question})
            self.conversation_history.append(
                {"role": "assistant", "content": response_trimmed}
            )

            self.is_first_turn = False
            self.last_state_code = "initialization"
            return response_trimmed

        state_code, extracted_memory = await self._track_state(question)

        if state_code == "E":
            self.last_state_code = "E"
            return ""

        self.last_state_code = state_code

        patient_response = await self._generate_response(
            question=question, state_code=state_code, extracted_memory=extracted_memory
        )

        # Update both short_term_memory (for state tracking) and conversation_history (for parent class)
        self.short_term_memory.append({"doctor": question, "patient": patient_response})
        self.conversation_history.append({"role": "user", "content": question})
        self.conversation_history.append(
            {"role": "assistant", "content": patient_response}
        )

        return patient_response

    def __name__(self, short: bool = True) -> str:
        return "StateAwarePatient"
