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

```python
!git clone -b automation https://github.com/Shravasti221/clinical_explainer.git
cd clinical_explainer
pip install -r "requirements.txt"
```

### Authentication
1. **Hugging Face Token**
   Login to Hugging Face and accept the terms for the required model.
   ```python
   from huggingface_hub import login
   import os
   login(token="<your hugging face token>")
   ```

2. **Groq API Key**
   Obtain your API key from [Groq Console](https://console.groq.com/keys).
   ```python
   os.environ["GROQ_API_KEY"] = "<GROQ API KEY>"
   ```

### Dataset Structure
Explore the CasiMedicos-Arg dataset:
```python
from datasets import load_dataset

dataset = load_dataset("HiTZ/casimedicos-exp", "en")
print(dataset["validation"].to_pandas().head())
```

## Usage: To generate token importances
```python
import torch
from clinical_explainer.src.strategies import ModelType
from clinical_explainer.src.pipeline import Pipeline

# Initialize pipeline for BERT model
pipeline = Pipeline(
    model_name="RUI525/PubMedBERT-finetune-MedMCQA-w-context",
    model_type=ModelType.ENCODER,  
    explainer_types=['lime', 'shap'],  
)

print("\n=== Starting Pipeline Processing ===")
print("Model: PubMedBERT-finetune-MedMCQA-w-context")
print("Processing first 12 validation cases")

# Process validation cases
results = pipeline.process_dataset(split='validation', limit=12)

# Collect save paths
save_paths = []

# Process results from generator
for idx, result in enumerate(results, 1):
    if 'error' not in result:
        lime_path = result['explanations'].get('lime', {}).get('save_dir')
        shap_path = result['explanations'].get('shap', {}).get('save_dir')
        
        if lime_path:
            save_paths.append(('lime', lime_path))
        if shap_path:
            save_paths.append(('shap', shap_path))
            
        print(f"\nCase {idx}:")
        print(f"LIME saved to: {lime_path}" if lime_path else "No LIME output")
        print(f"SHAP saved to: {shap_path}" if shap_path else "No SHAP output")
    else:
        print(f"\nCase {idx}: Error - {result['error']}")

# Print summary
print("\n=== Processing Summary ===")
print(f"Total cases processed: {len(save_paths)//2}") 
print("\nOutput Locations:")
for explainer_type, path in save_paths:
    print(f"{explainer_type.upper()}: {path}")

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

### Load prompt templates
```python
from clinical_explainer.src.utils.explanation_saver import load_lime_explanation, load_shap_explanation

# For LIME explanations:
lime_path = "path/to/lime/explanation"  
lime_data = load_lime_explanation(lime_path)

# For SHAP explanations:
shap_path = "path/to/shap/explanation"  
shap_data = load_shap_explanation(shap_path)

print("LIME Data:", lime_data)
print("SHAP Data:", shap_data)
```

### Print
```python
lime_exp = lime_data['explanation']

# Get feature weights and tokens
exp_list = lime_exp.as_list()  # Returns list of (token, weight) tuples

# Print in a readable format
print("Token Importance Values:")
for token, weight in exp_list:
    print(f"Token: {token:20} Weight: {weight:.4f}")
```
## Generate explanations based on the importance of tokens extracted from Phase.1
```python
import pandas as pd
import torch
from clinical_explainer.src.strategies import ModelType
from clinical_explainer.src.pipeline import Pipeline
from groq import Groq
import time
# Initialize pipeline for BERT model
model_name, model_type, explainer_types = "RUI525/PubMedBERT-finetune-MedMCQA-w-context", ModelType.ENCODER, ['shap', 'lime']
# model_name, model_type, explainer_types = "google/gemma-2-2b-it", ModelType.DECODER, ['TokenShap']
pipeline = Pipeline(
    model_name= model_name,
    model_type=model_type,  
    explainer_types=explainer_types 
)
client = Groq()

n= 1
number_of_tokens = 256

data_dict = {}
print("\n=== Starting Pipeline Processing ===")
print(f"Model: {model_name}")
print(f"Processing first {n} cases")

results = pipeline.process_dataset(
    split="validation",
    generate_natural_language_explanation=True,
    limit=n,
    explanation_max_tokens= number_of_tokens,
    llm_prompt_template = None # None or pass own prompt template to feed context rich data to LLM (Groq API used)
)

# Process each result
for case in results:
    question = case["original"]["full_question"]
    answer = case["original"]["full_answer"]
    correct_option = case["original"]["correct_option"]
    for explanation_type, explanation in case["context_grounded_reasoning"].items():
        key = (question, explanation_type, number_of_tokens)
        if key not in data_dict:
            data_dict[key] = {"correct_option": correct_option, "full_answer": answer}
        
        data_dict[key][f"predicted_explanation"] = explanation
                    
# Convert the dictionary into a DataFrame
data = pd.DataFrame.from_dict(data_dict, orient="index")
data.reset_index(inplace=True)
data.columns = ["question", "explanation_type", "max_tokens", "correct_option", "full_answer", "predicted_explanation"]


# Display the resulting DataFrame
print(data.head())
```
### Custom Prompt Templates
Customize the LLM prompts for generating explanations by constructing prompt templates as shown below.
```python
def prompt_function(user_prefix: str, question: str, prediction: str, explanation_text: str, user_suffix: str, assistant_prefix: str) -> str:
    """
    Constructs a detailed prompt for a language model to evaluate the token importances and explain predictions 
    for a medical multiple-choice question answering task.

    Args:
        user_prefix (str): Text to prepend before the question for user context.
        question (str): The medical question being analyzed.
        prediction (str): The predicted option selected by a smaller model.
        explanation_text (str): Explanation of the predicted option based on token importance scores.
        user_suffix (str): Text to append after the explanation to conclude the prompt.
        assistant_prefix (str): Text to signal the assistant's response or continuation.

    Returns:
        str: A formatted string combining the inputs to create a context-rich prompt for the model.
    """
    return (f" {user_prefix}\n"
            f"You are a Medical Expert. Evaluate the answer given by a model that is trained for answering medical questions. Explain why the answer selected is correct based on the question, available options, predicted option and the model's token importance scores.\n. \n"
            f"{question}" 
            f"OPTION PREDICTED by smaller model: {prediction}"
            f"{explanation_text}\n"
            f"{user_suffix}"
            f"{assistant_prefix}")
```
## Evaluation of Output:
The evaluation allows comparison of 2 texts or 2 columns of text in a dataframe to return a similarity score.
 
- **Embedding Similarity**: Based on the sentence transformer model selected from hugging face, returns pairwise cosine similarity of the embeddings of input text

- **NER Overlap Score**: Based on selected Token Classification model from hugging face, identifies the Named Entities in the 2 texts are returns 
\[
\text{NER Overlap (Jaccard Method)} = \frac{\text{Intersection(Named Entities in Text 1, Named Entities in Text 2)}}{\text{Union(Named Entities in Text 1, Named Entities in Text 2)}}
\]
- \[
\text{Weighted Score} = \text{NER\_Score} \times \text{Embedding\_similarity\_score}
\]

Assuming the below example dataframe
| Question                                                                 | Explanation Type | Max Tokens | Correct Option | Full Answer                                                                                                                                              | Prompt Type   | predicted_answer                                                                                           |
|--------------------------------------------------------------------------|------------------|------------|----------------|----------------------------------------------------------------------------------------------------------------------------------------------------------|---------------|-------------------------------------------------------------------------------------------------------|
| An 18-month-old boy, with complete immunization...                      | lime             | 256        | 4              | Although other hemorrhagic diseases can have a...                                                                                                         | context+eval  | Based on the given question, the most probable...                                                   |
| An 18-month-old boy, with complete immunization...                      | shap             | 256        | 4              | Although other hemorrhagic diseases can have a...                                                                                                         | context+eval  | The correct answer is option 4: Hemophilia A...                                                     |
| An 18-month-old boy, with complete immunization...                      | no_context       | 256        | 4              | Although other hemorrhagic diseases can have a...                                                                                                         | no_context    | The most probable diagnostic hypothesis is opt...                                                   |
| We are consulted by an 84-year-old woman for i...                       | lime             | 256        | 2              | Knowing that we have taken measures of sleep h...                                                                                                         | context+eval  | Based on the question, the patient is an 84-ye...                                                  |
| We are consulted by an 84-year-old woman for i...                       | shap             | 256        | 2              | Knowing that we have taken measures of sleep h...                                                                                                         | context+eval  | Based on the question, the patient is an 84-ye...                                                  |

### Code
```python
from clinical_explainer.src.evaluation import Evaluator
evaluations = Evaluator("all-MiniLM-L6-v2", "FacebookAI/xlm-roberta-large-finetuned-conll03-english")
df = evaluations.compute_dataframe_similarity(data_frame, "full_answer", "predicted_answer", "prefix_for_score_columns")
```


## Citation
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
