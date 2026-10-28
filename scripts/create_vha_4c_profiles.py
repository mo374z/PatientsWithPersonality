#!/usr/bin/env python3
"""Inject human-coded 4C red flags and contextual factors into the VHA patient profiles.

Each 4C row becomes one profile statement written into an existing profile field
(mapping by red-flag category in config), so every simulator receives identical information.
Each row also gets a clinician-observable cue (the red flag without its cause) for doctor briefings.
"""

import asyncio
import json
import logging
import sys
from pathlib import Path

import hydra
import pandas as pd
import pydantic
from omegaconf import DictConfig, OmegaConf
from tqdm.asyncio import tqdm as atqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from patient_simulator.misc.utils import create_llm_instance, retry_on_transient_error

log = logging.getLogger(__name__)

STATEMENT_PROMPT = (
    "Human coders annotated a real primary-care visit for a contextual red flag and, where "
    "the patient revealed it, the contextual factor behind it. Rewrite this annotation as one "
    "concise patient-profile statement in the third person (as in a chart), e.g. 'Often skips "
    "evening metformin because he works night shifts'. State only what the annotation "
    "supports; do not add details. If no factor is given, state only the problem.\n\n"
    "Red flag category: {category}\n"
    "Red flag: {red_flag}\n"
    "Contextual factor ({factor_category}): {factor}"
)


CUE_PROMPT = (
    "Human coders annotated a real primary-care visit for a contextual red flag: a sign that the "
    "patient's life context may be affecting their care. Rewrite the red flag as one concise "
    "cue a clinician could see in the chart or notice before talking to the patient (e.g. 'No "
    "metformin refills picked up for 3 months', 'HbA1c rose from 7.1 to 9.4'). Describe only the "
    "observable problem. Do not state or hint at any reason, cause, or circumstance behind it.\n\n"
    "Red flag category: {category}\n"
    "Red flag: {red_flag}"
)


class ClinicianCue(pydantic.BaseModel):
    cue: str


class ProfileStatement(pydantic.BaseModel):
    statement: str


def append_statement(profile: dict, field: str, statement: str) -> None:
    if field not in profile:
        raise KeyError(f"Profile has no field '{field}'")
    current = profile[field]
    profile[field] = (
        statement if current in ("Unknown", "None", "") else f"{current}; {statement}"
    )


async def run(cfg: DictConfig):
    output_dir = Path(cfg.output_dir)
    if output_dir.exists():
        raise FileExistsError(
            f"{output_dir} exists; statement generation is not deterministic, so regenerating "
            "would silently change profiles used by earlier runs. Choose a new output_dir."
        )
    cases = Path(cfg.case_list).read_text().split()
    four_c = pd.read_excel(cfg.four_c_path)
    four_c = four_c[four_c["Transcript ID"].isin(cases)]
    uncategorized = four_c["Red Flag Category"].isna()
    if uncategorized.any():
        log.warning(
            "Excluding %d red flags without a coded category", uncategorized.sum()
        )
        four_c = four_c[~uncategorized]
    unmapped = set(four_c["Red Flag Category"]) - set(cfg.red_flag_field)
    if unmapped:
        raise ValueError(
            f"Red flag categories without a field mapping: {sorted(unmapped)}"
        )

    writer = create_llm_instance(cfg.writer_model)
    sem = asyncio.Semaphore(int(cfg.concurrency))

    async def write(row):
        has_factor = isinstance(row["Factor Statements"], str)
        async with sem:
            result = await retry_on_transient_error(
                lambda: writer.generate_response(
                    prompt=STATEMENT_PROMPT.format(
                        category=row["Red Flag Category"],
                        red_flag=row["Red Flag Statements"],
                        factor_category=row["Factor Categories"]
                        if has_factor
                        else "none",
                        factor=row["Factor Statements"]
                        if has_factor
                        else "none documented",
                    ),
                    outlines_class=ProfileStatement,
                )
            )
        return result["response"].statement.strip().rstrip(".")

    async def write_cue(row):
        async with sem:
            result = await retry_on_transient_error(
                lambda: writer.generate_response(
                    prompt=CUE_PROMPT.format(
                        category=row["Red Flag Category"],
                        red_flag=row["Red Flag Statements"],
                    ),
                    outlines_class=ClinicianCue,
                )
            )
        return result["response"].cue.strip()

    rows = [row for _, row in four_c.iterrows()]
    statements = await atqdm.gather(*(write(row) for row in rows), desc="4C statements")
    cues = await atqdm.gather(*(write_cue(row) for row in rows), desc="Clinician cues")

    output_dir.mkdir(parents=True)
    injected = []
    for case in cases:
        profile = json.loads((Path(cfg.profiles_dir) / f"{case}.json").read_text())
        for flag_idx, (row, statement, cue) in enumerate(
            (r, s, c)
            for r, s, c in zip(rows, statements, cues)
            if r["Transcript ID"] == case
        ):
            field = cfg.red_flag_field[row["Red Flag Category"]]
            append_statement(profile, field, statement)
            injected.append(
                {
                    "case": case,
                    "flag_idx": flag_idx,
                    "category": row["Red Flag Category"],
                    "field": field,
                    "statement": statement,
                    "cue": cue,
                    "factor": row["Factor Statements"]
                    if isinstance(row["Factor Statements"], str)
                    else None,
                    "factor_category": row["Factor Categories"]
                    if isinstance(row["Factor Categories"], str)
                    else None,
                }
            )
        (output_dir / f"{case}.json").write_text(json.dumps(profile, indent=1))

    pd.DataFrame(injected).to_csv(output_dir / "injected_statements.csv", index=False)
    log.info("Wrote %d profiles with %d injected statements", len(cases), len(injected))


@hydra.main(
    version_base=None,
    config_path="../configs/experiment",
    config_name="vha_4c_profiles",
)
def main(cfg: DictConfig):
    log.info("Config:\n%s", OmegaConf.to_yaml(cfg))
    asyncio.run(run(cfg))


if __name__ == "__main__":
    main()
