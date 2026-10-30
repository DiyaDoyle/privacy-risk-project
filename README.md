# AI-Driven Intelligent Document Governance Framework with Layout-Aware Privacy Analysis

## Overview

This project presents an AI-driven document governance framework that combines document layout analysis, OCR-based text extraction, sensitive information detection, and layout-aware privacy risk assessment.

The framework investigates how the spatial relationships between sensitive entities in a document can support privacy risk analysis and governance recommendations.

## Key Features

- Document layout analysis
- OCR-based text extraction
- Personally Identifiable Information (PII) detection
- Layout-aware privacy risk scoring using LCPRS
- Structured JSON outputs for analysis and evaluation
- Evaluation datasets, ground-truth annotations, and result summaries

## Technology Stack

- Python
- PaddleOCR
- Document layout analysis models
- Sensitive information detection models
- JSON-based intermediate outputs
- Streamlit for the application interface

## Project Structure

```text
privacy-risk-project/
├── app.py
├── config/
│   ├── lcprs_config.json
│   └── pii_labels.json
├── evaluation/
│   ├── dataset/
│   ├── integrated_results/
│   ├── lcprs_results/
│   ├── phase2_results/
│   ├── stage1_ground_truth/
│   ├── stage1_results/
│   └── stage2_results/
├── layout_mapping.py
├── lcprs_v2.py
├── phase2_run_models.py
├── pipeline.py
├── process_new_document.py
├── stage1_mapping.py
├── requirements.txt
└── packages.txt
```

*Note: This shows the main project files and directories. Additional scripts, archived experiments, and evaluation outputs may also be present.*

## Installation and Setup

### Prerequisites

- Python
- Git
- Required dependencies for document layout analysis, OCR, and sensitive information detection

### 1. Clone the Repository

```bash
git clone https://github.com/DiyaDoyle/privacy-risk-project.git
cd privacy-risk-project
```

### 2. Create a Virtual Environment

```bash
python -m venv .venv
```

### 3. Activate the Virtual Environment

For Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

### 4. Install Dependencies

```bash
python -m pip install -r requirements.txt
```

### 5. Run the Application

If all required dependencies and models are configured, launch the Streamlit interface:

```bash
streamlit run app.py
```

## Evaluation

The `evaluation/` directory contains resources used to assess the framework, including:

- Dataset generation scripts
- Document annotations and ground-truth data
- Stage-wise processing results
- LCPRS risk assessment outputs
- Phase-wise evaluation results and summaries

These resources support the experimental evaluation of layout-aware privacy risk assessment.

## Research Objective

**Research Question:**

Can layout-aware document analysis improve the accuracy and effectiveness of privacy-aware document governance compared with conventional OCR-based document processing?

The project investigates whether document structure and spatial relationships can provide useful contextual information for privacy risk assessment beyond text extraction alone.

## Expected Outcomes

- Automated extraction and analysis of document content
- Identification of potentially sensitive information
- Layout-aware contextual privacy risk assessment
- Structured risk analysis outputs
- Experimental evaluation of the proposed framework

## Limitations

- Results depend on document quality, OCR extraction, layout detection, and sensitive information recognition.
- Risk scores are estimates and should be validated against annotated datasets.
- The framework is intended to support, not replace, professional privacy and compliance review.

## Disclaimer

This project is developed for academic research and experimentation. Privacy risk scores and governance recommendations should be validated before being used in real-world privacy or compliance decisions.

## Author

**Diya Mareeza Doyle**

