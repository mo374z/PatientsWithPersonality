from typing import Any
import json
import logging
import pandas as pd

from patient_simulator.misc.llm import LLM

log = logging.getLogger(__name__)


def _normalize_case_description(
    case_description: str | dict[str, Any],
) -> str | dict[str, Any]:
    """Parse the case description into a dict if possible"""
    if isinstance(case_description, str):
        try:
            case_description_dict = json.loads(case_description)
            if isinstance(case_description_dict, dict):
                return case_description_dict
        except json.JSONDecodeError:
            pass
    return case_description


class BasePatient:
    """Superclass for all Patient simulators.

    Provides common attributes and a shared teacher-forcing simulation utility
    that can be used across different patient implementations.
    """

    def __init__(
        self,
        case_description: str | dict[str, Any],
        llm: LLM,
    ):
        self.llm = llm
        self.case_description = _normalize_case_description(case_description)
        self.conversation_history: list[dict] = []

    async def get_response(self, question: str) -> str:  # pragma: no cover
        """Override in subclasses to generate a patient response."""
        raise NotImplementedError

    async def simulate_conversation(
        self,
        conv: list[tuple[str, str]],
        path: str | None = None,
        limit: int | None = None,
        max_retries: int = 3,
        align_doctor: bool = False,
    ) -> pd.DataFrame:
        """Run teacher-forcing simulation using a given conversation."""

        def _pair_doctor_patient(
            local_turns: list[tuple[str, str]],
        ) -> list[tuple[str, str]]:
            pairs: list[tuple[str, str]] = []
            i = 0
            n = len(local_turns)
            while i < n:
                speaker, text = local_turns[i]
                if speaker == "DOCTOR":
                    j = i + 1
                    if j < n and local_turns[j][0] == "PATIENT":
                        pairs.append((text, local_turns[j][1]))
                        i = j + 1
                    else:
                        i += 1
                elif speaker == "PATIENT":
                    pairs.append(("", text))
                    i += 1
                else:
                    raise ValueError(f"Unexpected speaker: {speaker}")
            return pairs

        pairs = _pair_doctor_patient(conv)
        if limit is not None:
            pairs = pairs[:limit]

        async def _align_doctor_response(
            history_turns: dict[int, dict[str, str]],
            response: str,
        ) -> str:
            patient_last_message = history_turns[max(history_turns)][
                "simulated_response"
            ]
            prompt = f"""Patient message:
{patient_last_message}

Original doctor response:
{response}

Task:
Edit the doctor response so it is consistent with the patient message.

Constraints:
- Make the smallest possible changes (minimal edits).
- Keep wording, structure, and length as close as possible to the original.
- Only modify parts that are inconsistent or fail to address the patient message.
- If the patient asks a question, answer it briefly within the existing response.
- Do not add new information or expand the response.

Return only the revised doctor response."""

            aligned_response = await self.llm.generate_response(prompt=prompt)
            if aligned_response["response"] is None:
                raise RuntimeError("LLM returned empty aligned doctor response")

            return aligned_response["response"].strip()

        full_conv = {}
        for idx, (doctor_q, real_resp) in enumerate(pairs):
            for attempt in range(max_retries):
                try:
                    doctor_q_aligned = doctor_q
                    if align_doctor and doctor_q and full_conv:
                        doctor_q_aligned = await _align_doctor_response(
                            history_turns=full_conv,
                            response=doctor_q,
                        )

                    sim_resp = await self.get_response(doctor_q_aligned)
                    break
                except Exception as e:
                    if attempt < max_retries - 1:
                        log.warning(
                            "Turn %d failed (attempt %d/%d): %s",
                            idx,
                            attempt + 1,
                            max_retries,
                            e,
                        )
                    else:
                        raise
            full_conv[idx] = {
                "doctor_question": doctor_q_aligned,
                "real_response": real_resp,
                "simulated_response": sim_resp,
            }
            if path is not None:
                pd.DataFrame.from_dict(full_conv, orient="index").to_csv(
                    f"{path}/turns.csv", index=False
                )

        return pd.DataFrame.from_dict(full_conv, orient="index")
