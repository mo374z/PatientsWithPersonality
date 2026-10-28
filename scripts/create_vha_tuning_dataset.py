#!/usr/bin/env python3
import argparse
import asyncio
import json
import logging
from itertools import product
from pathlib import Path
import random
import sys

import pandas as pd
import pydantic
from omegaconf import OmegaConf
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from patient_simulator.misc.style import style_profile
from patient_simulator.prompts.patient_prompts import FORM_DEFINITIONS
from patient_simulator.misc.utils import (
    create_llm_instance,
    parse_transcript_file,
    retry_on_transient_error,
)
from patient_simulator.patients.base_patient import pair_doctor_patient
from patient_simulator.patients.pwp import PatientsWithPersonality

log = logging.getLogger(__name__)

FORM_PROMPT = f"""Classify the form of this turn from a doctor in a clinical consultation.

{FORM_DEFINITIONS}

Doctor's turn:
{{question}}"""


PERSONA_GRID = [
    (dict(zip("hexaco", traits)), level)
    for traits in product((1, 2, 3), repeat=6)
    for level in ("A", "B", "C")
]


class FormLabel(pydantic.BaseModel):
    form: str


async def label_forms(pairs, llm, concurrency):
    semaphore = asyncio.Semaphore(concurrency)
    progress = tqdm(total=len(pairs), desc="Labelling turn forms")

    async def worker(index, question):
        async with semaphore:
            try:
                response = await retry_on_transient_error(
                    lambda: llm.generate_response(
                        prompt=FORM_PROMPT.format(question=question),
                        outlines_class=FormLabel,
                    )
                )
            except Exception as e:
                log.error("form labelling failed for turn %d: %s", index, e)
                progress.update(1)
                return index, None
        parsed = response["response"]
        label = (
            parsed.form
            if isinstance(parsed, FormLabel)
            else FormLabel.model_validate(parsed).form
        )
        progress.update(1)
        return index, label if label in {"polar", "open", "statement"} else None

    results = await asyncio.gather(*(worker(i, q) for i, (q, _) in enumerate(pairs)))
    progress.close()
    return dict(results)


def sample_stratified(indices_by_form, turns_per_case, rng):
    forms = [f for f in ("polar", "open", "statement") if indices_by_form.get(f)]
    if not forms:
        return []
    picked, per_form = [], max(1, turns_per_case // len(forms))
    for form in forms:
        pool = indices_by_form[form]
        picked += rng.sample(pool, min(per_form, len(pool)))
    remaining = [i for f in forms for i in indices_by_form[f] if i not in picked]
    rng.shuffle(remaining)
    return sorted(picked + remaining[: max(0, turns_per_case - len(picked))])


async def build(cfg):
    rng = random.Random(cfg["seed"])
    conversations_dir = Path(cfg["conversations_dir"])
    profiles_dir = Path(cfg["case_descriptions_dir"])

    cases = [
        c.strip() for c in Path(cfg["cases_file"]).read_text().split() if c.strip()
    ]
    llm = create_llm_instance(cfg["model_cfg"], keys_file=cfg["keys_file"])

    rows, targets = [], []
    for case in tqdm(cases, desc="Cases"):
        pairs = pair_doctor_patient(
            parse_transcript_file(str(conversations_dir / f"{case}.txt"))
        )
        if not pairs:
            raise ValueError(f"{case}: no doctor/patient pairs")
        forms = await label_forms(pairs, llm, cfg["concurrency"])

        indices_by_form = {}
        for i, form in forms.items():
            if form is not None:
                indices_by_form.setdefault(form, []).append(i)
        picked = sample_stratified(indices_by_form, cfg["turns_per_case"], rng)
        if not picked:
            raise ValueError(f"{case}: no turns could be labelled")

        profile = json.loads((profiles_dir / f"{case}.json").read_text())
        history = [
            {"doctor": pairs[i][0], "patient": pairs[i][1]} for i in range(len(pairs))
        ]
        turns_payload = json.dumps(
            [
                {
                    "index": i,
                    "doctor_question": pairs[i][0],
                    "real_response": pairs[i][1],
                    "form": forms[i],
                    "history": history[:i],
                }
                for i in picked
            ]
        )
        personas = rng.sample(PERSONA_GRID, cfg["configs_per_case"])
        for sample_idx, (traits, cefr_level) in enumerate(personas):
            patient = PatientsWithPersonality(
                case_description=profile,
                **traits,
                llm=llm,
                meta_llm=llm,
                level=cefr_level,
                register=cfg["register"],
                answer_form=cfg["answer_form"],
                dynamic_case_description=False,
                verbosity=0,
            )
            await patient._initialize_dynamic_prompts()

            rows.append(
                {
                    "row_id": f"{case}_cfg{sample_idx}",
                    "case_id": case,
                    "x": patient.build_role_inputs(),
                    "profile_json": json.dumps(profile),
                    "turns_json": turns_payload,
                    "cefr_level": cefr_level,
                    **{f"{axis}_trait": value for axis, value in traits.items()},
                }
            )
        real_sampled = [pairs[i][1] for i in picked]
        targets.append(
            {"case_id": case, "n_turns": len(picked), **style_profile(real_sampled)}
        )

    frame = pd.DataFrame(rows)
    duplicated = int(frame["x"].duplicated().sum())
    if duplicated:
        raise ValueError(
            f"{duplicated} rows share an identical 'x'; the reward task keys on it."
        )

    target_frame = pd.DataFrame(targets)
    shuffled = sorted(set(frame["case_id"]))
    rng.shuffle(shuffled)
    n_test = max(1, round(len(shuffled) * cfg["test_fraction"]))
    test_cases = set(shuffled[:n_test])
    frame["split"] = ["test" if c in test_cases else "train" for c in frame["case_id"]]
    target_frame["split"] = [
        "test" if c in test_cases else "train" for c in target_frame["case_id"]
    ]

    out = Path(cfg["output_path"])
    out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out, index=False)
    target_frame.to_csv(out.with_name(out.stem + "_targets.csv"), index=False)
    log.info(
        "%d cases (%d train / %d test), %d turns each -> %s",
        len(frame),
        (frame.split == "train").sum(),
        (frame.split == "test").sum(),
        cfg["turns_per_case"],
        out,
    )


def main():
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--conversations-dir", required=True)
    parser.add_argument("--case-descriptions-dir", required=True)
    parser.add_argument("--cases-file", required=True)
    parser.add_argument("--model-config", required=True)
    parser.add_argument("--keys-file", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--turns-per-case", type=int, required=True)
    parser.add_argument("--test-fraction", type=float, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--configs-per-case", type=int, required=True)
    parser.add_argument("--register", required=True, choices=["written", "spoken"])
    parser.add_argument("--answer-form", type=float, required=True)
    parser.add_argument("--concurrency", type=int, default=8)
    args = parser.parse_args()

    asyncio.run(
        build(
            {
                "conversations_dir": args.conversations_dir,
                "case_descriptions_dir": args.case_descriptions_dir,
                "cases_file": args.cases_file,
                "model_cfg": OmegaConf.to_container(
                    OmegaConf.load(args.model_config), resolve=True
                ),
                "keys_file": args.keys_file,
                "output_path": args.output_path,
                "turns_per_case": args.turns_per_case,
                "test_fraction": args.test_fraction,
                "seed": args.seed,
                "configs_per_case": args.configs_per_case,
                "register": args.register,
                "answer_form": args.answer_form,
                "concurrency": args.concurrency,
            }
        )
    )


if __name__ == "__main__":
    main()
