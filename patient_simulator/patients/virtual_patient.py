import json
from typing import Any

from patient_simulator.misc.llm import LLM
from patient_simulator.patients.base_patient import BasePatient
from patient_simulator.prompts.patient_prompts import (
    VIRTUAL_PATIENT_SYS,
)


class VirtualPatient(BasePatient):
    """Patient simulator following EasyMED's virtual patient behavior."""

    def __init__(
        self,
        case_description: str | dict[str, Any],
        llm: LLM,
    ):
        super().__init__(case_description=case_description, llm=llm)
        if not isinstance(self.case_description, dict):
            raise ValueError("VirtualPatient requires case_description to be a dict.")

    def _build_case_info(self) -> str:
        return json.dumps(self.case_description, indent=2, sort_keys=True)

    def _build_history_info(self) -> str:
        turns = []
        i = 0
        while i + 1 < len(self.conversation_history):
            user_msg = self.conversation_history[i]
            assistant_msg = self.conversation_history[i + 1]
            if user_msg["role"] == "user" and assistant_msg["role"] == "assistant":
                turns.append(
                    {
                        "question": user_msg["content"],
                        "answer": assistant_msg["content"],
                    }
                )
            i += 2

        if not turns:
            return ""

        history_lines = ["[Conversation History]"]
        for item in turns[-10:]:
            history_lines.append(f"Doctor asked: {item['question']}")
            history_lines.append(f"I answered: {item['answer']}")

        return "\n".join(history_lines)

    def _build_system_prompt(self) -> str:
        return VIRTUAL_PATIENT_SYS.format(
            case_info=self._build_case_info(),
            history_info=self._build_history_info(),
        )

    async def get_response(self, question: str) -> str:
        conversation_history = self.conversation_history + [
            {"role": "user", "content": question}
        ]

        response = await self.llm.generate_response(
            prompt=conversation_history,
            system_instruction=self._build_system_prompt(),
            sampling_kwargs={
                "temperature": 0.3,
                "stop": ["</response>", "Doctor:"],
            },
        )

        if response["response"] is None:
            raise RuntimeError("LLM returned empty response")

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
        return "VirtualPatient"
