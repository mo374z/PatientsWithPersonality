#!/usr/bin/env python3
"""Contextual-care benchmark on VHA cases with human-coded 4C ground truth.

A doctor model is briefed with the clinician-observable red flags of a visit, interviews a
simulated patient whose profile carries the coded contextual factors, and states for each red
flag the underlying contextual cause and a care plan. Scored against the coded factors:
contextual diagnosis accuracy (stated cause matches the factor), contextual error (plan does
not address the true cause), and agreement with the real physician's contextualized plan.
The real_transcript arm gives the same doctor model the real visit as the real-patient reference,
either in full, without the turns holding the real physician's coded care plan, or patient side only.
"""

import asyncio
import json
import logging
import random
import re
import sys
from pathlib import Path

import hydra
import pandas as pd
import pydantic
from omegaconf import DictConfig, ListConfig, OmegaConf
from tqdm.asyncio import tqdm as atqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from patient_simulator.misc.utils import create_llm_instance, retry_on_transient_error
from patient_simulator.patients import PATIENT_TYPE_MAP, PatientsWithPersonality

log = logging.getLogger(__name__)

END_INQUIRY_TOKEN = "[END_INQUIRY]"
VALIDATION_ATTEMPTS = 3

BRIEFING = (
    "You are a primary-care physician seeing a patient for a follow-up visit. Before the visit "
    "you reviewed the chart.\n\n"
    "Reason for visit: {visit_reason}\n"
    "Flagged issues:\n{cues}"
)

INTERVIEW_SYSTEM = (
    "{briefing}\n\n{instruction}\n\n"
    "Ask one concise question per turn. When you have gathered enough information, output the "
    f"token {END_INQUIRY_TOKEN} on its own line and stop asking questions."
)

ASSESSMENT_PROMPT = (
    "Flagged issues:\n{cues}\n\n"
    "Return exactly one entry per flagged issue, with its issue_number from the list above. For "
    "each, state whether you identified an underlying contextual cause (the circumstance in the "
    "patient's life, resources, beliefs, or behaviour that is behind it), what it is, and your "
    "care plan for that issue. Base the cause only on what the patient told you or the chart "
    "shows, not on guesses about what is typical."
)

CHART_PROMPT = "{briefing}\n\nFull chart:\n{chart}\n\n" + ASSESSMENT_PROMPT

CUES_ONLY_PROMPT = "{briefing}\n\n" + ASSESSMENT_PROMPT

REAL_TRANSCRIPT_PROMPT = (
    "{briefing}\n\nTranscript of your visit with the patient{note}:\n{conversation}\n\n"
    + ASSESSMENT_PROMPT
)

TRANSCRIPT_NOTES = {
    "full": "",
    "without_care_plan": " (the clinician's care-plan statements are removed)",
    "patient_only": " (only what the patient and companions said)",
}

SPEAKER_HEADER = re.compile(
    r"^(DOCTOR|MEDICAL STUDENT|PATIENT|SECONDE? PERSON|THIRD PERSON)[ \w]*:?[ \t]*$",
    re.M,
)

SPEAKER_ROLES = {
    "DOCTOR": "assistant",
    "MEDICAL STUDENT": "assistant",
    "PATIENT": "patient",
}

SPAN = re.compile(r"\((\d+),\s*(\d+)\)")

CONTEXT_JUDGE_PROMPT = (
    "Human coders identified the contextual factor behind each red flag below in a real "
    "primary-care visit. A physician later assessed the same issues. For each item decide:\n"
    "- cause_correct: the physician's stated contextual cause identifies the same underlying "
    "circumstance, barrier, or belief as the coded factor, even if phrased more generally (e.g. "
    "'misunderstood the follow-up instructions' matches 'thought she had to go back to the ED for "
    "removal'). A different barrier, a purely biomedical explanation, a list of possible "
    "barriers, or 'not identified' is not a match.\n"
    "- plan_addresses_cause: the physician's plan for this issue addresses the coded factor "
    "(adapts care to that circumstance), regardless of what cause the physician stated.\n"
    "- plan_matches_real: the physician's plan takes the same contextual approach as the real "
    "physician's care plan statements quoted for the item. False if none are quoted.\n\n"
    "{items}\n\n"
    "Return one entry per item, in input order."
)

ELICITATION_JUDGE_PROMPT = (
    "Human coders identified the contextual factors below in a real primary-care visit. For each "
    "factor, decide whether the patient in the conversation below expresses the same factor "
    "(same circumstance, barrier, or belief; wording may differ).\n\n"
    "Factors:\n{factors}\n\n"
    "Conversation:\n{conversation}\n\n"
    "Return exactly {n} booleans, one per numbered factor, in input order."
)


class IssueAssessment(pydantic.BaseModel):
    issue_number: int
    cause_identified: bool
    contextual_cause: str
    plan: str


class Assessment(pydantic.BaseModel):
    issues: list[IssueAssessment]


class ContextJudgment(pydantic.BaseModel):
    cause_correct: bool
    plan_addresses_cause: bool
    plan_matches_real: bool


class ContextJudgments(pydantic.BaseModel):
    entries: list[ContextJudgment]


class Elicitation(pydantic.BaseModel):
    expressed: list[bool]


async def validated_call(llm, prompt, schema, is_valid, requirement: str):
    for attempt in range(VALIDATION_ATTEMPTS):
        response = (
            await retry_on_transient_error(
                lambda: llm.generate_response(
                    prompt=prompt, outlines_class=schema, use_cache=attempt == 0
                )
            )
        )["response"]
        if is_valid(response):
            return response
        log.warning(
            "Invalid output (attempt %d), expected %s", attempt + 1, requirement
        )
    raise ValueError(
        f"Invalid output after {VALIDATION_ATTEMPTS} attempts, expected {requirement}"
    )


async def length_checked_call(llm, prompt, schema, expected: dict[str, int]):
    return await validated_call(
        llm,
        prompt,
        schema,
        lambda r: {field: len(getattr(r, field)) for field in expected} == expected,
        f"lengths {expected}",
    )


def sample_params(params) -> dict:
    return {
        k: random.choice(list(v)) if isinstance(v, (list, ListConfig)) else v
        for k, v in params.items()
    }


def create_patient(ptype, params, case_description, patient_llm, meta_llm):
    if ptype not in PATIENT_TYPE_MAP:
        raise ValueError(f"Unknown patient config type: {ptype}")
    patient_class = PATIENT_TYPE_MAP[ptype]
    extra_kwargs = (
        {"meta_llm": meta_llm} if patient_class is PatientsWithPersonality else {}
    )
    return patient_class(case_description, llm=patient_llm, **extra_kwargs, **params)


async def run_inquiry(patient, assistant_llm, max_turns: int, system_instruction: str):
    history: list[dict] = []
    turns: list[dict] = []
    patient_msg = await patient.get_response(
        "Please describe what is bothering you today."
    )
    turns.append({"role": "patient", "content": patient_msg})
    history.append({"role": "user", "content": patient_msg})

    n_turns = 0
    for _ in range(max_turns):
        result = await assistant_llm.generate_response(
            prompt=history, system_instruction=system_instruction
        )
        assistant_msg = result["response"].strip()
        turns.append({"role": "assistant", "content": assistant_msg})
        history.append({"role": "assistant", "content": assistant_msg})
        n_turns += 1
        if END_INQUIRY_TOKEN.lower() in assistant_msg.lower():
            break
        patient_msg = await patient.get_response(assistant_msg)
        turns.append({"role": "patient", "content": patient_msg})
        history.append({"role": "user", "content": patient_msg})

    return history, turns, n_turns


def numbered(items: list[str]) -> str:
    return "\n".join(f"{i + 1}. {item}" for i, item in enumerate(items))


def coded_rows(four_c: pd.DataFrame, case: str) -> pd.DataFrame:
    rows = four_c[
        (four_c["Transcript ID"] == case) & four_c["Red Flag Category"].notna()
    ]
    return rows.reset_index(drop=True)


def coded_spans(cell) -> list[tuple[int, int]]:
    return (
        [(int(a), int(b)) for a, b in SPAN.findall(cell)]
        if isinstance(cell, str)
        else []
    )


def parse_raw_transcript(text: str) -> list[dict]:
    headers = list(SPEAKER_HEADER.finditer(text))
    turns = []
    for header, nxt in zip(headers, headers[1:] + [None]):
        end = nxt.start() if nxt else len(text)
        content = " ".join(text[header.end() : end].split())
        if content:
            role = SPEAKER_ROLES.get(header.group(1), "companion")
            turns.append(
                {"role": role, "content": content, "span": (header.end(), end)}
            )
    return turns


def load_real_turns(path: Path, plan_spans: list[tuple[int, int]]) -> list[dict]:
    turns = parse_raw_transcript(path.read_text(encoding="utf-8", errors="replace"))
    unmatched = [
        p
        for p in plan_spans
        if not any(p[0] < t["span"][1] and t["span"][0] < p[1] for t in turns)
    ]
    if unmatched:
        raise ValueError(
            f"{path}: care-plan spans {unmatched} fall outside every parsed turn"
        )
    for t in turns:
        t["care_plan"] = any(
            p[0] < t["span"][1] and t["span"][0] < p[1] for p in plan_spans
        )
    return turns


def select_real_turns(turns: list[dict], transcript: str) -> list[dict]:
    if transcript == "full":
        return turns
    if transcript == "without_care_plan":
        return [t for t in turns if not t["care_plan"]]
    if transcript == "patient_only":
        return [t for t in turns if t["role"] != "assistant"]
    raise ValueError(f"Unknown transcript variant: {transcript}")


def attach_real_coding(flags: list[dict], rows: pd.DataFrame, case: str) -> None:
    if len(rows) != len(flags):
        raise ValueError(
            f"{case}: {len(flags)} injected flags vs {len(rows)} coded 4C rows"
        )
    for f, (_, row) in zip(flags, rows.iterrows()):
        if row["Red Flag Category"] != f["category"] or pd.isna(
            row["Factor Statements"]
        ) != pd.isna(f["factor"]):
            raise ValueError(
                f"{case} flag {f['flag_idx']}: injected statements and 4C rows misaligned"
            )
        f["real_probed"] = isinstance(row["Probe Statements"], str)
        f["real_care_plan"] = (
            row["Care Plan Statements"]
            if isinstance(row["Care Plan Statements"], str)
            else None
        )


def load_cases(cfg: DictConfig) -> list[dict]:
    statements = pd.read_csv(cfg.statements_path)
    four_c = pd.read_excel(cfg.four_c_path)
    cases = []
    for case in Path(cfg.case_list).read_text().split():
        profile = json.loads((Path(cfg.profiles_dir) / f"{case}.json").read_text())
        flags = (
            statements[statements.case == case]
            .sort_values("flag_idx")
            .to_dict("records")
        )
        rows = coded_rows(four_c, case)
        attach_real_coding(flags, rows, case)
        plan_spans = [
            span for cell in rows["Care Plan Offsets"] for span in coded_spans(cell)
        ]
        cues = numbered([f["cue"] for f in flags])
        cases.append(
            {
                "id": case,
                "profile": profile,
                "flags": flags,
                "cues": cues,
                "real_turns": load_real_turns(
                    Path(cfg.transcripts_dir) / f"{case}.txt", plan_spans
                ),
                "briefing": BRIEFING.format(
                    visit_reason=profile["chiefcomplaint"], cues=cues
                ),
            }
        )
    return cases


def factor_flags(case: dict) -> list[dict]:
    return [f for f in case["flags"] if isinstance(f["factor"], str)]


def format_conversation(turns: list[dict]) -> str:
    labels = {"patient": "PATIENT", "assistant": "DOCTOR", "companion": "COMPANION"}
    return "\n".join(f"{labels[t['role']]}: {t['content']}" for t in turns)


async def assess(llm, prompt, n_issues: int) -> dict[int, IssueAssessment]:
    expected = list(range(1, n_issues + 1))
    response = await validated_call(
        llm,
        prompt,
        Assessment,
        lambda r: sorted(i.issue_number for i in r.issues) == expected,
        f"issue numbers {expected}",
    )
    return {i.issue_number - 1: i for i in response.issues}


async def judge_context(
    judge_llm, case: dict, assessment: dict[int, IssueAssessment]
) -> list[ContextJudgment]:
    items = [
        f"Item {i + 1}\n  red flag: {f['cue']}\n  coded factor ({f['factor_category']}): "
        f"{f['factor']}\n  physician's stated cause: {assessment[f['flag_idx']].contextual_cause}"
        f"\n  physician's plan: {assessment[f['flag_idx']].plan}"
        f"\n  real physician's care plan statements: {f['real_care_plan'] or 'none quoted'}"
        for i, f in enumerate(factor_flags(case))
    ]
    response = await length_checked_call(
        judge_llm,
        CONTEXT_JUDGE_PROMPT.format(items="\n\n".join(items)),
        ContextJudgments,
        {"entries": len(items)},
    )
    return response.entries


async def judge_elicitation(judge_llm, case: dict, turns: list[dict]) -> list[bool]:
    flags = factor_flags(case)
    response = await length_checked_call(
        judge_llm,
        ELICITATION_JUDGE_PROMPT.format(
            factors=numbered(
                [
                    f"[{f['factor_category']}] {' … '.join(f['factor'].split(' *** '))}"
                    for f in flags
                ]
            ),
            conversation=format_conversation(turns),
            n=len(flags),
        ),
        Elicitation,
        {"expressed": len(flags)},
    )
    return response.expressed


def flag_row(
    flag: dict, issue: IssueAssessment, judgment: ContextJudgment | None, elicited
) -> dict:
    row = {
        "flag_idx": flag["flag_idx"],
        "red_flag_category": flag["category"],
        "factor_category": flag["factor_category"],
        "has_factor": judgment is not None,
        "real_probed": flag["real_probed"],
        "real_plan_contextualized": flag["real_care_plan"] is not None,
        "cause_identified": issue.cause_identified,
        "stated_cause": issue.contextual_cause,
        "cause_correct": None,
        "contextual_error": None,
        "plan_matches_real": None,
        "factor_elicited": elicited,
    }
    if judgment is not None:
        row["cause_correct"] = judgment.cause_correct
        row["contextual_error"] = not judgment.plan_addresses_cause
        if flag["real_care_plan"] is not None:
            row["plan_matches_real"] = judgment.plan_matches_real
    return row


async def run_arm(arm, case, assistant_llm, patient):
    n_issues = len(case["flags"])
    if arm.type == "chart":
        prompt = CHART_PROMPT.format(
            briefing=case["briefing"],
            chart=json.dumps(case["profile"], indent=1),
            cues=case["cues"],
        )
        return await assess(assistant_llm, prompt, n_issues), None, 0
    if arm.type == "cues_only":
        prompt = CUES_ONLY_PROMPT.format(briefing=case["briefing"], cues=case["cues"])
        return await assess(assistant_llm, prompt, n_issues), None, 0
    if arm.type == "real_transcript":
        turns = select_real_turns(case["real_turns"], arm.transcript)
        prompt = REAL_TRANSCRIPT_PROMPT.format(
            briefing=case["briefing"],
            note=TRANSCRIPT_NOTES[arm.transcript],
            conversation=format_conversation(turns),
            cues=case["cues"],
        )
        return await assess(assistant_llm, prompt, n_issues), turns, len(turns) // 2
    if arm.type != "interview":
        raise ValueError(f"Unknown doctor arm type: {arm.type}")
    history, turns, n_turns = await run_inquiry(
        patient,
        assistant_llm,
        arm.max_turns,
        INTERVIEW_SYSTEM.format(briefing=case["briefing"], instruction=arm.instruction),
    )
    prompt = history + [
        {"role": "user", "content": ASSESSMENT_PROMPT.format(cues=case["cues"])}
    ]
    return await assess(assistant_llm, prompt, n_issues), turns, n_turns


async def _run(cfg: DictConfig):
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cases = [c for c in load_cases(cfg) if factor_flags(c)]
    log.info(
        "Loaded %d cases with %d coded factors",
        len(cases),
        sum(len(factor_flags(c)) for c in cases),
    )

    patient_llm = create_llm_instance(cfg.patient_model, use_cache=False)
    meta_llm = create_llm_instance(cfg.meta_model, use_cache=False)
    judge_llm = create_llm_instance(cfg.judge_model, use_cache=True)

    results_path = output_dir / "results.csv"
    completed = set()
    if results_path.exists():
        existing = pd.read_csv(results_path, dtype={"case_id": str})
        completed = set(
            zip(
                existing.assistant_model,
                existing.doctor_arm,
                existing.patient_name,
                existing.case_id,
                existing.run,
            )
        )
    header_written = results_path.exists()
    sem = asyncio.Semaphore(int(cfg.concurrency))
    write_lock = asyncio.Lock()

    async def run_one(model_name, assistant_llm, arm, case, pc, run_idx):
        nonlocal header_written
        if pc is None:
            patient, patient_name, params = None, arm.name, {}
        else:
            params = sample_params(pc.get("params") or {})
            patient = create_patient(
                pc["patient_type"], params, dict(case["profile"]), patient_llm, meta_llm
            )
            patient_name = patient.__name__(short=False)
        if (model_name, arm.name, patient_name, case["id"], run_idx) in completed:
            return

        async with sem:
            try:
                assessment, turns, n_turns = await run_arm(
                    arm, case, assistant_llm, patient
                )
                judgments, expressed = await asyncio.gather(
                    judge_context(judge_llm, case, assessment),
                    judge_elicitation(judge_llm, case, turns)
                    if turns
                    else asyncio.sleep(0),
                )
            except Exception as e:
                log.warning(
                    "Run failed (%s, %s, %s, %s): %s",
                    model_name,
                    arm.name,
                    patient_name,
                    case["id"],
                    e,
                )
                return

        run_dir = output_dir / patient_name / model_name / arm.name / case["id"]
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / f"assessment_{run_idx}.json").write_text(
            json.dumps(
                {k: v.model_dump() for k, v in sorted(assessment.items())}, indent=2
            )
        )
        if turns:
            pd.DataFrame(turns)[["role", "content"]].to_csv(
                run_dir / f"turns_{run_idx}.csv", index=False
            )

        expressed = expressed if turns else [None] * len(judgments)
        judged = {
            f["flag_idx"]: (j, e)
            for f, j, e in zip(factor_flags(case), judgments, expressed)
        }
        run_info = {
            "assistant_model": model_name,
            "doctor_arm": arm.name,
            "patient_name": patient_name,
            "persona_params": json.dumps(params),
            "case_id": case["id"],
            "run": run_idx,
            "n_turns": n_turns,
        }
        rows = [
            run_info
            | flag_row(
                f, assessment[f["flag_idx"]], *judged.get(f["flag_idx"], (None, None))
            )
            for f in case["flags"]
        ]
        async with write_lock:
            pd.DataFrame(rows).to_csv(
                results_path, mode="a", header=not header_written, index=False
            )
            header_written = True

    patient_configs = OmegaConf.to_container(cfg.patient_configs, resolve=True)
    try:
        for asst_cfg in cfg.assistant_models:
            model_name = asst_cfg.get("full_name") or asst_cfg.name
            assistant_llm = create_llm_instance(asst_cfg, use_cache=False)
            tasks = [
                run_one(model_name, assistant_llm, arm, case, pc, run_idx)
                for arm in cfg.doctor_arms
                for pc in (patient_configs if arm.type == "interview" else [None])
                for case in cases
                for run_idx in range(cfg.n_runs)
            ]
            await atqdm.gather(*tasks, desc=f"Visits ({model_name})")
    finally:
        if hasattr(meta_llm, "cleanup"):
            meta_llm.cleanup()

    df = pd.read_csv(results_path)
    summary = (
        df.groupby(["assistant_model", "doctor_arm", "patient_name"])[
            [
                "cause_correct",
                "contextual_error",
                "plan_matches_real",
                "factor_elicited",
                "n_turns",
            ]
        ]
        .mean()
        .round(3)
    )
    summary["cause_identified_no_factor"] = (
        df[~df.has_factor]
        .groupby(["assistant_model", "doctor_arm", "patient_name"])
        .cause_identified.mean()
        .round(3)
    )
    summary.to_csv(output_dir / "summary.csv")
    print(summary.to_string())


@hydra.main(
    version_base=None,
    config_path="../configs/experiment",
    config_name="contextual_care_vha",
)
def main(cfg: DictConfig):
    asyncio.run(_run(cfg))


if __name__ == "__main__":
    main()
