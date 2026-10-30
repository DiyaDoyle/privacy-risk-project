\# AI-Driven Intelligent Document Governance Framework with Layout-Aware Privacy Analysis



\## Overview



This project presents an AI-driven document governance framework that combines document layout analysis, OCR-based text extraction, sensitive information detection, and layout-aware privacy risk assessment.



The framework investigates how the spatial relationships between sensitive entities in a document can support privacy risk analysis and governance recommendations.



\## Key Features



\- Document layout analysis

\- OCR-based text extraction

\- Personally identifiable information (PII) detection

\- Layout-aware privacy risk scoring using LCPRS

\- Structured JSON outputs for analysis and evaluation

\- Evaluation datasets, ground-truth annotations, and result summaries



\## Technology Stack



\- Python

\- PaddleOCR and document layout analysis models

\- Sensitive information detection models

\- JSON-based intermediate outputs

\- Streamlit for the application interface



\## Project Structure



\- `app.py` — application interface

\- `config/` — configuration files and PII labels

\- `evaluation/` — evaluation datasets, annotations, scripts, and results

\- `layout\_mapping.py` — document layout mapping

\- `lcprs\_v2.py` — LCPRS risk assessment logic

\- `pipeline.py` — processing pipeline

\- `phase2\_run\_models.py` — Phase 2 evaluation workflow

\- `process\_new\_document.py` — document processing workflow

\- `stage1\_mapping.py` — Stage 1 mapping

\- `requirements.txt` — Python dependencies

\- `packages.txt` — additional package configuration



\## Setup



1\. Clone this repository.

2\. Create and activate a Python virtual environment.

3\. Install the dependencies:



&#x20;  ```bash

&#x20;  pip install -r requirements.txt

&#x20;  ```



4\. Review the configuration files in `config/`.

5\. Run the application locally:



&#x20;  ```bash

&#x20;  streamlit run app.py

&#x20;  ```



\## Evaluation



The `evaluation/` directory contains dataset-generation scripts, annotations, ground-truth data, and evaluation outputs used to assess the framework. Consult the relevant scripts and result summaries for the evaluation workflow.



\## Research Focus



Can layout-aware document analysis improve the accuracy and effectiveness of privacy-aware document governance compared with conventional OCR-based document processing?



\## Disclaimer



This project is intended for academic research and experimentation. Risk scores and governance recommendations should be validated before being used in real-world privacy decisions.



