```markdown
# Clinical Decision Support Explainer

A model-agnostic pipeline for analyzing and explaining multiple-choice medical question answering models using LIME and SHAP.

## Overview
This project provides tools for processing and explaining predictions from medical multiple-choice question answering models. It is designed to work with the CasiMedicos-Arg dataset and various HuggingFace models fine-tuned for medical question answering.

## Dataset
- **CasiMedicos-Arg**: A dataset of 558 clinical cases in multiple languages
  - 434 training cases
  - 63 validation cases
  - 125 test cases
- Each case includes:
  - Clinical scenario
  - Multiple choice options
  - Correct answer
  - Expert explanations

## Features
- Model-agnostic pipeline for multiple-choice medical QA
- Support for any HuggingFace multiple-choice model
- Detailed prediction outputs with confidence scores
- Memory-efficient processing using generators
- Explainability using LIME and SHAP (coming soon)

## Installation

Install required packages:
```python
pip install datasets transformers torch
```

## Usage
```python
from src import Pipeline

# Initialize pipeline with your chosen model
pipeline = Pipeline("RUI525/PubMedBERT-finetune-MedMCQA-w-context")
pipeline.setup()

# Process cases
results = pipeline.process_dataset('validation', limit=10)
```

Example output:
```
Loading CasiMedicos-Arg validation split...
Processing 10 cases...

================================================================================
📋 Case ID: 274
Type: DIGESTIVE SYSTEM

📝 Question:
A 52-year-old man with no concomitant diseases comes to the emergency department...

🔤 Options:
Option 1: In the initial endoscopy...
Option 2: In the initial endoscopy, given the absence...
Option 3: In initial endoscopy, endoscopic therapy...
Option 4: Since this is a complicated ulcer...

🎯 Prediction Results:
Model predicted: Option 1
Correct answer: Option 1
Status: ✅ Correct
Confidence: 42.77%
```

## Project Structure
```
clinical_explainer/
└── src/
    ├── __init__.py
    ├── pipeline.py
    └── explainers/
        ├── __init__.py
        ├── lime_explainer.py
        └── shap_explainer.py
```

## Requirements
- Python 3.7+
- PyTorch
- Transformers
- Datasets
```
## License
MIT License