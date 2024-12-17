# Clinical Decision Support Explainer

A model-agnostic pipeline for analyzing and explaining multiple-choice medical question answering models using LIME and SHAP.

## Overview
This project provides tools for processing and explaining predictions from medical multiple-choice question answering models. It is designed to work with the CasiMedicos-Arg dataset and various HuggingFace models fine-tuned for medical question answering.

## Dataset
**CasiMedicos-Arg**: A dataset of 558 clinical cases in multiple languages
  - 434 training cases
  - 63 validation cases
  - 125 test cases

Dataset Links:
- [CasiMedicos-Arg on HuggingFace](https://huggingface.co/datasets/HiTZ/casimedicos-exp)

## Features
- Model-agnostic pipeline for multiple-choice medical QA
- Support for any HuggingFace multiple-choice model
- Detailed prediction outputs with confidence scores
- Memory-efficient processing using generators
- Explainability using LIME and SHAP with visualization
- Save and load explanations for later analysis


## Installation

Install required packages:
```python
pip install datasets transformers torch lime
```

## Usage
```python
from clinical_explainer.src.pipeline import Pipeline
from clinical_explainer.src.explainers import ExplainerType

# Initialize pipeline with LIME explainer
pipeline = Pipeline(
    model_name="RUI525/PubMedBERT-finetune-MedMCQA-w-context",
    explainer_types=[ExplainerType.LIME, ExplainerType.SHAP]
)
pipeline.setup()

# Process cases and get explanations
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

## Requirements
- Python 3.7+
- PyTorch
- Transformers
- Datasets
- Lime
- Shap

Citation:
```bibtex
@inproceedings{otegi-etal-2023-casimedicos,
    title = "{C}asi{M}edicos-Arg: A Dataset for Argumentative Clinical Case Resolution and Explanation in Medical Education",
    author = "Otegi, Arantxa  and
      Campos, Jon Ander  and
      Agirre, Eneko",
    booktitle = "Proceedings of the 18th Workshop on Innovative Use of NLP for Building Educational Applications (BEA 2023)",
    month = jul,
    year = "2023",
    address = "Toronto, Canada",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.bea-1.5",
    pages = "44--55"
}
@misc{goenaga2023explanatory,
      title={Explanatory Argument Extraction of Correct Answers in Resident Medical Exams}, 
      author={Iakes Goenaga and Aitziber Atutxa and Koldo Gojenola and Maite Oronoz and Rodrigo Agerri},
      year={2023},
      eprint={2312.00567},
      archivePrefix={arXiv}
}
