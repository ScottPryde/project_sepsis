import json
import sys
import threading
from pathlib import Path
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd

from sepsis_prediction.application import MODEL_IDS, REVIEW_JOB, create_application_server
from sepsis_prediction.cli import main
from sepsis_prediction.data import PatientRecord
from sepsis_prediction.data_review import run_data_review
from sepsis_prediction.features import create_examples
from sepsis_prediction.quality import AGE_TOP_CODE_RECODED, apply_quality_rules


def record(patient_id: str, source: str = "A", **columns) -> PatientRecord:
    length = len(next(iter(columns.values())))
    frame = pd.DataFrame({"SepsisLabel": [0] * length, "ICULOS": list(range(1, length + 1)), **columns})
    return PatientRecord(patient_id, frame.astype(float), source)


def by_rule(audit: dict) -> dict:
    return {rule["rule_id"]: rule for rule in audit["rules"]}


def test_range_rule_removes_impossible_values_only() -> None:
    records = [record("p1", HR=[80, 400, 90], FiO2=[0.5, 4000, 0.04], Potassium=[4.0, 27.5, np.nan])]

    cleaned, audit = apply_quality_rules(records)

    frame = cleaned[0].frame
    assert frame["HR"].tolist()[0::2] == [80, 90] and np.isnan(frame.loc[1, "HR"])
    assert frame["FiO2"].isna().tolist() == [False, True, True]
    assert np.isnan(frame.loc[1, "Potassium"])
    assert by_rule(audit)["Q2_plausible_range"]["values_changed_by_field"] == {"HR": 1, "FiO2": 2, "Potassium": 1}
    assert frame["SepsisLabel"].tolist() == [0, 0, 0] and frame["ICULOS"].tolist() == [1, 2, 3]


def test_ionised_calcium_is_attributed_to_its_own_rule() -> None:
    cleaned, audit = apply_quality_rules([record("p1", Calcium=[8.5, 1.12, 9.0, 0.2])])

    assert cleaned[0].frame["Calcium"].isna().tolist() == [False, True, False, True]
    rules = by_rule(audit)
    assert rules["Q1_calcium_ionised"]["values_changed"] == 1
    assert rules["Q2_plausible_range"]["values_changed_by_field"] == {"Calcium": 1}


def test_inverted_pressure_clears_that_row_only() -> None:
    cleaned, audit = apply_quality_rules([record("p1", SBP=[120, 70], DBP=[70, 90], MAP=[85, 80])])

    frame = cleaned[0].frame
    assert frame.loc[0, ["SBP", "DBP", "MAP"]].tolist() == [120, 70, 85]
    assert frame.loc[1, ["SBP", "DBP", "MAP"]].isna().all()
    assert by_rule(audit)["Q3_inverted_pressure"]["rows_affected"] == 1


def test_bilirubin_and_age_rules() -> None:
    cleaned, _ = apply_quality_rules([record(
        "p1", Bilirubin_direct=[0.5, 3.0], Bilirubin_total=[1.0, 2.0], Age=[100.0, 100.0],
    )])

    frame = cleaned[0].frame
    assert frame["Bilirubin_direct"].isna().tolist() == [False, True]
    assert frame["Age"].tolist() == [AGE_TOP_CODE_RECODED] * 2


def test_lab_carry_forward_keeps_first_of_each_hourly_run() -> None:
    records = [
        record("p1", BUN=[20, 20, 20, 25, np.nan, 25], FiO2=[0.4, 0.4, 0.4, 0.4, 0.4, 0.4], HR=[80] * 6),
        record("p2", BUN=[25, 25]),
    ]

    cleaned, audit = apply_quality_rules(records)

    assert cleaned[0].frame["BUN"].tolist()[:4] == [20, None, None, 25] or (
        cleaned[0].frame["BUN"].isna().tolist() == [False, True, True, False, True, False]
    )
    assert cleaned[0].frame["FiO2"].notna().all()
    assert cleaned[0].frame["HR"].notna().all()
    assert cleaned[1].frame["BUN"].tolist() == [25, None] or cleaned[1].frame["BUN"].isna().tolist() == [False, True]
    assert by_rule(audit)["Q6_lab_carry_forward"]["values_changed_by_field"] == {"BUN": 3}


def test_cleaning_preserves_shape_and_cohort_and_none_is_a_no_op() -> None:
    rng = np.random.default_rng(1)
    records = [
        record(f"p{index}", "A" if index % 2 else "B", HR=list(rng.normal(85, 8, 30)),
               SepsisLabel=[0] * (6 + index % 5) + [1] * (24 - index % 5))
        for index in range(12)
    ]
    records[0].frame.loc[3, "HR"] = 999.0

    cleaned, audit = apply_quality_rules(records)
    untouched, none_audit = apply_quality_rules(records, "none")

    assert [len(item.frame) for item in cleaned] == [len(item.frame) for item in records]
    assert [list(item.frame.columns) for item in cleaned] == [list(item.frame.columns) for item in records]
    assert create_examples(cleaned)[1].equals(create_examples(records)[1])
    assert audit["values_changed"] >= 1 and by_rule(audit)["Q2_plausible_range"]["patients_affected_by_source"] == {"B": 1}
    assert untouched is records and none_audit == {"ruleset": "none", "rules": []}


def write_review_cohort(root: Path) -> list[str]:
    rng = np.random.default_rng(5)
    folders = []
    for source, offset in (("training_setA", 0), ("training_setB", 1)):
        folder = root / source
        folder.mkdir(parents=True)
        folders.append(str(folder))
        for number in range(20):
            hours = 30
            labels = [0] * hours
            if number % 4 == 0:
                labels[8:] = [1] * (hours - 8)
            frame = pd.DataFrame({
                "HR": rng.normal(85 + 10 * offset, 8, hours),
                "SBP": rng.normal(120, 10, hours), "DBP": rng.normal(65, 8, hours), "MAP": rng.normal(85, 8, hours),
                "Calcium": np.where(np.arange(hours) % 6 == 0, 8.5 if offset == 0 else 1.1, np.nan),
                "BUN": np.repeat(rng.normal(18, 3, hours // 3).round(), 3),
                "Age": 100.0 if number == 0 else 60.0, "Gender": number % 2,
                "SepsisLabel": labels, "ICULOS": np.arange(1, hours + 1),
            })
            if number == 1:
                frame.loc[4, "HR"] = 500
            frame.to_csv(folder / f"p{offset}{number:05d}.psv", sep="|", index=False)
    return folders


def test_data_review_page_and_findings(tmp_path: Path, monkeypatch) -> None:
    folders = write_review_cohort(tmp_path / "raw")
    output_dir = tmp_path / "review"
    monkeypatch.setattr(sys, "argv", [
        "sepsis-pipeline", "review", "--data-dir", *folders, "--output-dir", str(output_dir),
    ])
    main()

    review = json.loads((output_dir / "data_review.json").read_text(encoding="utf-8"))
    assert review["dataset"]["patients"] == 40
    assert review["fields"]["HR"]["implausible"]["above"] == 2
    assert {item["value"] for item in review["fields"]["HR"]["implausible_examples"]} == {500}
    assert "implausible values" in review["fields"]["HR"]["flags"]
    assert review["fields"]["Calcium"]["by_source"]["training_setB"]["median"] < 2
    impact = {rule["rule_id"]: rule for rule in review["rule_impact"]}
    assert impact["Q1_calcium_ionised"]["patients_affected_by_source"] == {"training_setB": 20}
    assert impact["Q5_age_top_code"]["patients_affected"] == 2
    assert impact["Q6_lab_carry_forward"]["values_changed_by_field"]["BUN"] > 0
    assert any(item["priority"] == "Implement" for item in review["recommendations"])
    assert (output_dir / "field_summary.csv").is_file()
    assert (output_dir / "charts" / "field_HR.png").is_file()
    page = (output_dir / "data_review.html").read_text(encoding="utf-8")
    for text in ("Introduction", "Tests and their rationale", "Where the review sits in the pipeline", "<svg",
                 "Recommendations", 'id="field-HR"', "<details", "Cross-field consistency", "Record-level integrity"):
        assert text in page, text
    assert "https://" not in page


def test_run_applies_rules_and_records_them(tmp_path: Path, monkeypatch) -> None:
    folders = write_review_cohort(tmp_path / "raw")
    output_dir = tmp_path / "run"
    monkeypatch.setattr(sys, "argv", [
        "sepsis-pipeline", "run", "--data-dir", *folders, "--output-dir", str(output_dir),
        "--models", "logistic_regression", "--horizon-hours", "6",
    ])
    main()

    audit = json.loads((output_dir / "quality_audit.json").read_text(encoding="utf-8"))
    config = json.loads((output_dir / "run_config.json").read_text(encoding="utf-8"))
    assert audit["ruleset"] == "standard" and audit["values_changed"] > 0
    assert config["quality"]["ruleset"] == "standard"
    assert [rule["rule_id"] for rule in config["quality"]["rules"]][0] == "Q1_calcium_ionised"
    assert "Data quality rules (standard)" in (output_dir / "report.html").read_text(encoding="utf-8")


def test_application_serves_review_page_and_summary(tmp_path: Path) -> None:
    review_dir = tmp_path / "data_review"
    commands = {model: [sys.executable, "-c", "pass"] for model in (*MODEL_IDS, REVIEW_JOB)}
    model_dirs = {model: tmp_path / "models" / model for model in MODEL_IDS} | {REVIEW_JOB: review_dir}
    server = create_application_server(
        tmp_path, tmp_path / "report.html", model_dirs, commands, tmp_path / "fallback",
        Path(__file__).parents[1] / "src" / "sepsis_prediction" / "web", False, port=0, review_dir=review_dir,
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base_url = f"http://127.0.0.1:{server.server_port}"
    try:
        with urlopen(base_url + "/data-review.html") as response:
            assert b"No data review yet" in response.read()
        with urlopen(base_url + "/api/review") as response:
            assert json.loads(response.read())["available"] is False
        with urlopen(base_url + "/") as response:
            assert b"Open data review" in response.read()

        review_dir.mkdir()
        (review_dir / "data_review.html").write_text("<h1>Review</h1>", encoding="utf-8")
        (review_dir / "data_review.json").write_text(json.dumps({
            "dataset": {"patients": 3}, "recommendations": [{}],
            "fields": {"HR": {"flags": ["implausible values"], "implausible": {"below": 0, "above": 2, "not_allowed": 0}}},
            "rule_impact": [{"rule_id": "Q2_plausible_range", "title": "range", "values_changed": 2, "patients_affected": 1}],
        }), encoding="utf-8")
        with urlopen(base_url + "/data-review.html") as response:
            assert response.read() == b"<h1>Review</h1>"
        with urlopen(base_url + "/api/review") as response:
            payload = json.loads(response.read())
        assert payload["available"] is True and payload["fields_flagged"] == 1 and payload["implausible_values"] == 2
        assert payload["can_run"] is True and payload["status"]["state"] == "complete"
        with urlopen(Request(base_url + f"/api/run/{REVIEW_JOB}", method="POST", data=b"")) as response:
            assert response.status == 202
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
