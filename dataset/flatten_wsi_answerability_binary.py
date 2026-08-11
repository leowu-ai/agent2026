#!/usr/bin/env python3
import argparse
import json
import re
from pathlib import Path
from collections import Counter

def norm(s):
    return re.sub(r"\s+", " ", str(s or "").strip().lower())

def semantic_label(item):
    q = norm(item.get("Question"))
    choices = [norm(x) for x in item.get("Choice", [])]

    # Clinical / report / procedural / follow-up metadata.
    false_terms = [
        "patient's age", "patient age", "chief complaint", "symptom", "performance scale",
        "karnofsky", "menopausal", "birth control", "consent", "who signed",
        "treatment", "medication", "recommended treatment", "recommended next step",
        "recommendations", "procedure did", "surgical procedures", "pre-operative diagnosis",
        "preoperative diagnosis", "clinical diagnosis", "what stains were performed",
        "stains were performed", "what additional tests", "tests were performed",
        "tests are pending", "additional tests are pending", "pending immunohistochemistry",
        "specimen format", "anatomic site", "which sections are included",
        "where were receptor studies performed", "what information is not provided",
        "comments or notes", "what does the slide include information on",
    ]
    if any(t in q for t in false_terms):
        return False, "requires_clinical_report_or_procedural_context"

    # Follow-up outcomes are not recoverable from the WSI itself.
    if "vital_status" in q or "vital status" in q or "survival time" in q or "survival_time" in q:
        return False, "requires_followup_information"

    # Actual assay-specific measurements/results.
    if "fish" in q or "gene amplification" in q or "amplification observed" in q:
        return False, "requires_actual_assay_result"
    if any(t in q for t in ["percentage of nuclear staining", "percent nuclear staining",
                            "percentage of neoplastic cell nuclei", "proliferation index",
                            "score for estrogen receptors"]):
        return False, "requires_exact_assay_measurement"

    # Exact physical measurements / specimen-level quantification.
    quantitative_terms = [
        "size of the tumor", "tumor size", "size of the mass", "size of the lump",
        "size of the invasive", "size of the infiltr", "size of the breast tumor",
        "size of the largest", "size was", "tumor diameter", "diameter of the tumor",
        "measurements of the tumor", "measurement of the biopsy cavity",
        "size of the labeled", "what size was", "how big is", "distance between",
        "closest margin", "depth of the closest margin", "how many foci",
        "what percentage", "percentage of", "how much of the tumor",
        "percentage of the tumor", "percentage of carcinoma", "percentage of tumor",
    ]
    if any(t in q for t in quantitative_terms):
        return False, "requires_exact_case_level_measurement"

    # Staging is case-level, not a pure WSI visual answer.
    stage_terms = [
        "pathological stage", "pathologic stage", "final pathological stage",
        "final pathologic stage", "stage classification", "stage of the tumor",
        "cancer stage", "stage of breast cancer", "ajcc staging", "tnm stage",
        "tnm classification", "ptnm staging", "tumor stage", "staging of the tumor",
        "what stage is the tumor"
    ]
    if any(t in q for t in stage_terms):
        return False, "requires_case_level_staging_context"

    # Margin / orientation / laterality / specimen location.
    margin_terms = [
        "surgical margin", "surgical margins", "margin assessment", "margin of resection",
        "resection margin", "margin of excision", "margins involved", "margins free",
        "margin(s)", "medial new margin", "dorsal resection margin",
        "deep surgical margin", "where in the patient's breast",
        "which breast was scheduled", "in which breast did", "where was metastasis detected",
        "where was metastatic involvement", "what type of node", "sentinel node #",
        "superior mastectomy flap", "previous biopsy site"
    ]
    if any(t in q for t in margin_terms):
        return False, "requires_specimen_orientation_or_case_context"

    # Categorical ER/PR/HER2 phenotype is considered WSI-predictable.
    receptor_terms = [
        "estrogen receptor", "progesterone receptor", "hormone receptor",
        "hormonal receptor", "receptor status", "her2", "her-2", "her 2-neu",
        "her2 neu", "er, pr", "er/pr"
    ]
    if any(t in q for t in receptor_terms):
        return True, "categorical_wsi_predictable_biological_phenotype"

    # H&E morphology.
    morphology_terms = [
        "diagnosis", "histological type", "histologic type", "histological_type",
        "type of carcinoma", "type of cancer", "type of tumor", "subtype",
        "grade", "differentiated", "nuclear grade", "nottingham score",
        "nottingham histologic score", "pattern", "necrosis", "microcalcification",
        "microcalcifications", "lymphovascular", "vascular invasion", "angioinvasion",
        "in-situ component", "in situ component", "dcis", "lcis",
        "hyperplasia", "fibrocystic", "fibroadenoma", "adenosis", "metaplasia",
        "pathological finding", "histological finding", "histopathological",
        "microscopic appearance", "features observed", "tissue changes",
        "additional findings", "other findings", "lesion was found",
        "presence of breast cancer", "what was found in the breast",
        "what is found in the surrounding breast tissue", "invasive status",
        "invasion present", "invasion identified"
    ]
    if any(t in q for t in morphology_terms):
        return True, "wsi_morphology"

    return None, "no_high_priority_rule"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset_json")
    ap.add_argument("binary_spec_json")
    ap.add_argument("output_json")
    args = ap.parse_args()

    data = json.loads(Path(args.dataset_json).read_text(encoding="utf-8"))
    spec = json.loads(Path(args.binary_spec_json).read_text(encoding="utf-8"))
    mc = [x for x in data if x.get("Choice", x.get("choices"))]

    exact = {(x["Id"], x["Question"]): x for x in spec["exact_labels"]}
    rows = []
    seen = set()
    for item in mc:
        key = (str(item.get("Id", "")), str(item.get("Question", "")))
        if key in seen:
            raise ValueError(f"Duplicate MC key: {key}")
        seen.add(key)

        if key in exact:
            e = exact[key]
            can_answer = e["can_answer"]
            exclude = bool(e.get("exclude_from_evaluation", False))
            reason = e.get("reason_code", "exact_label")
            source = "exact_label"
        else:
            inferred, reason = semantic_label(item)
            if inferred is None:
                can_answer = bool(spec.get("default_for_remaining_mc_items", True))
                source = "remaining_mc_default"
            else:
                can_answer = inferred
                source = "semantic_rule"
            exclude = False

        rows.append({
            "Id": key[0],
            "Question": key[1],
            "can_answer": can_answer,
            "exclude_from_evaluation": exclude,
            "reason_code": reason,
            "label_source": source
        })

    expected = int(spec["source_dataset"]["multiple_choice_expected_total"])
    if len(rows) != expected:
        raise ValueError(f"Expected {expected} MC items, got {len(rows)}")

    counts = Counter(
        "excluded" if r["exclude_from_evaluation"] else ("can_answer" if r["can_answer"] else "cannot_answer")
        for r in rows
    )
    output = {
        "schema_version": "wsi_answerability_binary_flat_v1",
        "source_dataset": spec["source_dataset"],
        "summary": dict(counts),
        "labels": rows
    }
    Path(args.output_json).write_text(
        json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({"total": len(rows), **dict(counts)}, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
