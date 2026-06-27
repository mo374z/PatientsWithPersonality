from typing import Any

from patient_simulator.misc.llm import LLM
from patient_simulator.patients.base_patient import BasePatient


class RealPatient(BasePatient):
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
        """Return the exact real patient response for the current question without rephrasing."""
        if self.response_index >= len(self.real_responses):
            return "[ERROR] No more real responses available to rephrase"

        # Get the real response for this turn
        real_response = self.real_responses[self.response_index]
        self.response_index += 1

        return real_response

    def __name__(self, short: bool = True) -> str:
        return "RealPatient"
