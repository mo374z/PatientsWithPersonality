from typing import Any

from patient_simulator.misc.llm import LLM
from patient_simulator.patients.base_patient import BasePatient


class BaselinePatient(BasePatient):
    """Baseline patient simulator that rephrases real patient answers.

    This patient uses the real patient responses from the conversation
    transcript and simply rephrases them using an LLM. This provides a
    baseline for comparison with more sophisticated patient simulators.
    """

    def __init__(
        self,
        case_description: str | dict[str, Any],
        llm: LLM,
        real_responses: list[str] | None = None,
    ):
        super().__init__(case_description, llm.model)
        self.llm = llm
        self.real_responses = real_responses or []
        self.response_index = 0

    async def get_response(self, question: str) -> str:
        """Rephrase the real patient response for the current question.

        Args:
            question: The doctor's question

        Returns:
            Rephrased version of the real patient response
        """
        if self.response_index >= len(self.real_responses):
            return "[ERROR] No more real responses available to rephrase"

        # Get the real response for this turn
        real_response = self.real_responses[self.response_index]
        self.response_index += 1

        system_instruction = """You are a helpful assistant that rephrases patient responses. Maintain the same meaning, information content, and tone."""

        prompt = f"""Rephrase the following patient response. Keep the response length similar to the original response:\n\n {real_response}\n\n Provide your answer to the doctor's inquiry between the tags <response> and </response>. Do not include any other text in your answer."""

        response = await self.llm.generate_response(
            prompt=prompt,
            system_instruction=system_instruction,
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

        self.conversation_history.append({"role": "user", "content": question})
        self.conversation_history.append(
            {"role": "assistant", "content": response_trimmed}
        )

        return response_trimmed

    def __name__(self, short: bool = True) -> str:
        return "BaselinePatient"
