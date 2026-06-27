from typing import Any

from patient_simulator.misc.llm import LLM
from patient_simulator.patients.base_patient import (
    _normalize_case_description,
    BasePatient,
)
from patient_simulator.prompts.patient_prompts import (
    AGENTCLINIC_BIASES,
    AGENTCLINIC_SYS,
)


class AgentClinicPatient(BasePatient):
    """Patient simulator following AgentClinic's approach (Schmidgall et al., 2024)"""

    def __init__(
        self,
        case_description: str | dict[str, Any],
        llm: LLM,
        bias_present: str = None,
    ):
        self.llm = llm
        self.case_description = _normalize_case_description(case_description)
        self.bias_present = bias_present
        self.conversation_history = []
        self.system_instruction = self._build_system_prompt()

    def _build_system_prompt(self) -> str:
        """Build the system prompt following AgentClinic's approach"""
        bias_prompt = (
            AGENTCLINIC_BIASES.get(self.bias_present, "") if self.bias_present else ""
        )
        return AGENTCLINIC_SYS.format(
            bias=bias_prompt, case_description=self.case_description
        )

    async def get_response(self, question: str) -> str:
        """Get patient response to doctor's question"""
        prompt = f"""\nHere is a history of your dialogue: {
            self._format_conversation_history()
        }\n This is the latest doctor's inquiry: {question}\n
            Provide your answer to the doctor's inquiry between the tags <response> and </response>. Do not include any other text in your answer."""

        response = await self.llm.generate_response(
            prompt=prompt,
            system_instruction=self.system_instruction,
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

        self.conversation_history.append({"role": "user", "content": question})
        self.conversation_history.append(
            {"role": "assistant", "content": response_trimmed}
        )

        return response_trimmed

    def _format_conversation_history(self) -> str:
        """Format conversation history as a string for the prompt"""
        if not self.conversation_history:
            return ""

        history_str = ""
        for msg in self.conversation_history:
            if msg["role"] == "user":
                history_str += f"Doctor: {msg['content']}\n"
            else:
                history_str += f"Patient: {msg['content']}\n"

        return history_str

    def __name__(self, short: bool = True) -> str:
        if short:
            return "AgentClinicPatient"
        return (
            f"AgentClinicPatient_{self.bias_present}"
            if self.bias_present
            else "AgentClinicPatient_nobias"
        )
