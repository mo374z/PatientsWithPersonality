from typing import Any
from patient_simulator.misc.llm import LLM
from patient_simulator.patients.base_patient import (
    BasePatient,
    _normalize_case_description,
)

from patient_simulator.prompts.patient_prompts import (
    CRAFTMD_SYS,
)


class CraftMDPatient(BasePatient):
    """Patient simulator following CraftMD's approach (Johri et al., 2023)"""

    def __init__(
        self,
        case_description: str | dict[str, Any],
        llm: LLM,
    ):
        self.llm = llm
        self.conversation_history = []
        normalized_case = _normalize_case_description(case_description)
        self.system_instruction = CRAFTMD_SYS.format(case_desc=normalized_case)

    async def get_response(self, question: str) -> str:
        conversation_history = self.conversation_history + [
            {"role": "user", "content": question}
        ]

        response = await self.llm.generate_response(
            prompt=conversation_history,
            system_instruction=self.system_instruction,
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
        conversation_history.append({"role": "assistant", "content": response_trimmed})

        self.conversation_history = conversation_history
        return response_trimmed

    def __name__(self, short: bool = True) -> str:
        return "CraftMDPatient"
