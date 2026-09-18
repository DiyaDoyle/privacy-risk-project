import os
import sys
import json
import subprocess
from pathlib import Path


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent


# ============================================================
# EXISTING PIPELINE SCRIPTS
# ============================================================

STAGE1_SCRIPT = PROJECT_ROOT / "stage1_mapping.py"
STAGE2_SCRIPT = PROJECT_ROOT / "layout_mapping.py"
LCPRS_SCRIPT = PROJECT_ROOT / "lcprs_v2.py"


# ============================================================
# OUTPUT DIRECTORIES
# ============================================================

STAGE1_OUTPUT = (
    PROJECT_ROOT
    / "evaluation"
    / "stage1_results"
)

STAGE2_OUTPUT = (
    PROJECT_ROOT
    / "evaluation"
    / "stage2_results"
)

LCPRS_OUTPUT = (
    PROJECT_ROOT
    / "evaluation"
    / "lcprs_results"
)

INTEGRATED_OUTPUT = (
    PROJECT_ROOT
    / "evaluation"
    / "integrated_results"
)


# ============================================================
# CREATE OUTPUT DIRECTORY
# ============================================================

INTEGRATED_OUTPUT.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# PYTHON EXECUTABLE
# ============================================================

PYTHON = sys.executable


# ============================================================
# VALIDATE REQUIRED FILES
# ============================================================

required_scripts = [
    STAGE1_SCRIPT,
    STAGE2_SCRIPT,
    LCPRS_SCRIPT
]


for script in required_scripts:

    if not script.exists():

        print()
        print("=" * 70)
        print("ERROR")
        print("=" * 70)

        print(
            f"Required script not found:\n{script}"
        )

        print("=" * 70)

        raise SystemExit(1)


# ============================================================
# RUN STAGE
# ============================================================

def run_stage(
    stage_name,
    script_path
):

    print()
    print("=" * 70)
    print(stage_name)
    print("=" * 70)

    print()
    print(
        f"Running: {script_path.name}"
    )

    print()

    result = subprocess.run(
        [
            PYTHON,
            str(script_path)
        ],
        cwd=str(PROJECT_ROOT)
    )

    if result.returncode != 0:

        print()
        print("=" * 70)
        print(f"{stage_name} FAILED")
        print("=" * 70)

        print(
            f"Exit code: {result.returncode}"
        )

        raise SystemExit(
            result.returncode
        )

    print()
    print(
        f"{stage_name} completed successfully."
    )


# ============================================================
# STAGE 1
# ============================================================

run_stage(
    "STAGE 1 - PII DETECTION AND OCR MAPPING",
    STAGE1_SCRIPT
)


# ============================================================
# STAGE 2
# ============================================================

run_stage(
    "STAGE 2 - PII TO LAYOUT MAPPING",
    STAGE2_SCRIPT
)


# ============================================================
# LCPRS
# ============================================================

run_stage(
    "STAGE 3 - LCPRS RISK SCORING",
    LCPRS_SCRIPT
)


# ============================================================
# CREATE INTEGRATED SUMMARY
# ============================================================

print()
print("=" * 70)
print("CREATING INTEGRATED RESULTS")
print("=" * 70)


lcprs_files = sorted(
    LCPRS_OUTPUT.glob(
        "*_lcprs.json"
    )
)


if not lcprs_files:

    print()
    print(
        "ERROR: No LCPRS result files found."
    )

    print(
        f"Expected results inside:\n{LCPRS_OUTPUT}"
    )

    raise SystemExit(1)


summary = []


for result_file in lcprs_files:

    with open(
        result_file,
        "r",
        encoding="utf-8"
    ) as file:

        data = json.load(file)


    document = data.get(
        "document",
        result_file.stem
    )

    number_of_pii = data.get(
        "number_of_pii",
        data.get(
            "pii_items_used",
            len(
                data.get(
                    "pii_items",
                    []
                )
            )
        )
    )

    score = data.get(
        "normalized_lcprs_score"
    )

    category = data.get(
        "risk_category"
    )

    governance = data.get(
        "governance_action"
    )


    summary.append(
        {
            "document": document,
            "number_of_pii": number_of_pii,
            "lcprs_score": score,
            "risk_category": category,
            "governance_action": governance,
            "result_file": str(
                result_file.relative_to(
                    PROJECT_ROOT
                )
            )
        }
    )


# ============================================================
# SAVE SUMMARY
# ============================================================

summary_file = (
    INTEGRATED_OUTPUT
    / "pipeline_summary.json"
)


with open(
    summary_file,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        {
            "pipeline": [
                "PaddleOCR",
                "GLiNER2-PII",
                "OCR word-level mapping",
                "PP-DocLayout-L",
                "PII-to-layout mapping",
                "LCPRS V2"
            ],
            "documents": summary
        },
        file,
        indent=4,
        ensure_ascii=False
    )


# ============================================================
# PRINT FINAL SUMMARY
# ============================================================

print()

print(
    "Integrated pipeline results:"
)

print()

for item in summary:

    print(
        f"Document       : "
        f"{item['document']}"
    )

    print(
        f"PII count      : "
        f"{item['number_of_pii']}"
    )

    print(
        f"LCPRS score    : "
        f"{item['lcprs_score']:.2f}"
        if isinstance(
            item["lcprs_score"],
            (int, float)
        )
        else
        f"LCPRS score    : "
        f"{item['lcprs_score']}"
    )

    print(
        f"Risk category  : "
        f"{item['risk_category']}"
    )

    print(
        f"Governance     : "
        f"{item['governance_action']}"
    )

    print(
        f"Result         : "
        f"{item['result_file']}"
    )

    print("-" * 70)


# ============================================================
# FINAL OUTPUT LOCATIONS
# ============================================================

print()
print("=" * 70)
print("INTEGRATED PIPELINE COMPLETED")
print("=" * 70)

print()

print(
    "Stage 1 results:"
)

print(
    STAGE1_OUTPUT
)

print()

print(
    "Stage 2 visual + JSON results:"
)

print(
    STAGE2_OUTPUT
)

print()

print(
    "LCPRS results:"
)

print(
    LCPRS_OUTPUT
)

print()

print(
    "Integrated summary:"
)

print(
    summary_file
)

print()

print("=" * 70)
print("ALL PIPELINE STAGES COMPLETED")
print("=" * 70)