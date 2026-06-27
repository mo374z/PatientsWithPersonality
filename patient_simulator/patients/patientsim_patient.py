from patient_simulator.misc.llm import LLM
from patient_simulator.patients.base_patient import BasePatient
from patient_simulator.prompts.patient_prompts import (
    PATIENTSIM_INITIAL_SYS,
    PATIENTSIM_CEFR,
    PATIENTSIM_DAZED,
    PATIENTSIM_PERSONALITY,
    PATIENTSIM_RECALL,
    PATIENTSIM_SENTENCELENGTH,
    PATIENTSIM_WORD_LISTS,
)


class PatientSimPatient(BasePatient):
    """Patient simulator following PatientSim's approach (Kyung et al., 2025)"""

    def __init__(
        self,
        patient_profile: dict[str, str],
        llm: LLM,
        cefr_type: str,
        personality_type: str,
        recall_level_type: str,
        dazed_level_type: str,
        num_word_sample: int = 3,
    ):
        """Create a PatientSim patient from structured profile dict.

        Valid options:
        - cefr_type: A, B, C
        - personality_type: plain, verbose, pleasing, impatient, distrust, overanxious
        - recall_level_type: low, high
        - dazed_level_type: normal, moderate, high
        """
        self.llm = llm
        self.patient_profile = patient_profile.copy()
        self.conversation_history = []
        self.num_word_sample = num_word_sample

        self.cefr_type = cefr_type
        self.personality_type = personality_type
        self.recall_level_type = recall_level_type
        self.dazed_level_type = dazed_level_type

        self._validate_arguments()
        self._build_profile_attributes()
        self.system_instruction = self._build_system_prompt()

    def _validate_arguments(self) -> None:
        """Validate that all persona settings are valid."""
        valid_cefr = list(PATIENTSIM_CEFR.keys())
        valid_personality = list(PATIENTSIM_PERSONALITY.keys())
        valid_recall = list(PATIENTSIM_RECALL.keys())
        valid_dazed = list(PATIENTSIM_DAZED.keys())

        if self.cefr_type not in valid_cefr:
            raise ValueError(
                f"Invalid CEFR type: {self.cefr_type}. Choose one of: {', '.join(valid_cefr)}"
            )
        if self.personality_type not in valid_personality:
            raise ValueError(
                f"Invalid personality type: {self.personality_type}. Choose one of: {', '.join(valid_personality)}"
            )
        if self.recall_level_type not in valid_recall:
            raise ValueError(
                f"Invalid recall level type: {self.recall_level_type}. Choose one of: {', '.join(valid_recall)}"
            )
        if self.dazed_level_type not in valid_dazed:
            raise ValueError(
                f"Invalid dazed level type: {self.dazed_level_type}. Choose one of: {', '.join(valid_dazed)}"
            )

    def _build_profile_attributes(self) -> None:
        """Build and format persona attributes for prompt injection."""
        cefr_levels = ["A", "B", "C"]
        current_index = cefr_levels.index(self.cefr_type)
        higher_level = cefr_levels[current_index + 1] if self.cefr_type != "C" else None

        self.patient_profile["understand_words"] = ", ".join(
            PATIENTSIM_WORD_LISTS.get(f"cefr_{self.cefr_type}1", "").split(", ")[
                : self.num_word_sample
            ]
        )
        self.patient_profile["misunderstand_words"] = ", ".join(
            PATIENTSIM_WORD_LISTS.get(f"cefr_{self.cefr_type}2", "").split(", ")[
                : self.num_word_sample
            ]
        )
        self.patient_profile["understand_med_words"] = ", ".join(
            PATIENTSIM_WORD_LISTS.get(f"med_{self.cefr_type}", "").split(", ")[
                : self.num_word_sample
            ]
        )
        self.patient_profile["misunderstand_med_words"] = (
            ", ".join(
                PATIENTSIM_WORD_LISTS.get(f"med_{higher_level}", "").split(", ")[
                    : self.num_word_sample
                ]
            )
            if higher_level is not None
            else ""
        )

        self.patient_profile["cefr"] = PATIENTSIM_CEFR[self.cefr_type].format(
            **self.patient_profile
        )
        self.patient_profile["personality"] = PATIENTSIM_PERSONALITY[
            self.personality_type
        ]

        if self.personality_type != "plain":
            self.patient_profile["personality"] += (
                "\n\tIMPORTANT: Ensure that your personality is clearly represented throughout the conversation, "
                "while allowing your emotional tone and style to vary naturally across turns."
            )

        self.patient_profile["memory_recall_level"] = PATIENTSIM_RECALL[
            self.recall_level_type
        ]

        self._build_dazed_level()
        self._build_reminder()

        self.patient_profile["sent_limit"] = PATIENTSIM_SENTENCELENGTH.get(
            self.personality_type, "3"
        )

    def _build_dazed_level(self) -> None:
        """Build the dazed level description with progressive phases."""
        if self.dazed_level_type == "normal":
            self.patient_profile["dazed_level"] = PATIENTSIM_DAZED["normal"]
        else:
            dazed_levels = ["high", "moderate", "normal"]
            dazed_states = ["initial", "intermediate", "later"]
            dazed_index = dazed_levels.index(self.dazed_level_type)

            dazed_description = (
                f"\nThe patient's initial dazed level is {self.dazed_level_type}. "
                "The dazedness should gradually fade throughout the conversation as the doctor continues to reassure them. "
                "Transitions should feel smooth and natural, rather than abrupt. "
                "While the change should be subtle and progressive, the overall dazed level is expected to decrease noticeably every 4-5 turns, "
                "following the instructions for each level below."
            )

            for _dazed_index in range(dazed_index, len(dazed_levels)):
                dazed_text = PATIENTSIM_DAZED[dazed_levels[_dazed_index]]
                dazed_description += (
                    f"\n{dazed_levels[_dazed_index].capitalize()} Dazedness ({dazed_states[_dazed_index].capitalize()} Phase)\n\t\t"
                    + "\n\t\t".join(dazed_text.split("\n")[1:])
                )

            dazed_description += "\n\tNote: Dazedness reflects the patient's state of confusion and inability in following the conversation, independent of their language proficiency."
            self.patient_profile["dazed_level"] = dazed_description

    def _build_reminder(self) -> None:
        """Build a concise reminder of the patient's persona."""
        cefr_first_line = PATIENTSIM_CEFR[self.cefr_type].split("\n")[0]
        personality_first_line = PATIENTSIM_PERSONALITY[self.personality_type].split(
            "\n"
        )[0]
        recall_first_line = PATIENTSIM_RECALL[self.recall_level_type].split("\n")[0]
        dazed_first_line = PATIENTSIM_DAZED[self.dazed_level_type].split("\n")[0]

        self.patient_profile["reminder"] = (
            f"You should act like {cefr_first_line}. You are {personality_first_line}. "
            f"Also, you {recall_first_line.lower()}. {dazed_first_line}"
        )

    def _build_system_prompt(self) -> str:
        """Build the system prompt from the initial template and patient profile."""

        class _UnknownDefaultDict(dict):
            def __missing__(self, key):
                return "Unknown"

        return PATIENTSIM_INITIAL_SYS.format_map(
            _UnknownDefaultDict(self.patient_profile)
        )

    async def get_response(self, question: str) -> str:
        """Get patient response to doctor's question."""
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
        if short:
            return "PatientSimPatient"
        return f"PatientSimPatient_pers{self.personality_type}_cefr{self.cefr_type}_dazed{self.dazed_level_type}_recall{self.recall_level_type}"
