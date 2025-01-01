import pandas as pd
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import numpy as np

def calculate_baseline_similarity(text_corpus: pd.Series, sample_size=1000, model_name='all-MiniLM-L6-v2', random_state=None):
    """
    Calculates a baseline similarity score for text within a column of a DataFrame.

    Args:
        dataframe (pd.DataFrame): Input DataFrame containing the text column.
        text_column (str): Name of the column containing text data.
        sample_size (int): Number of random pairs to sample for similarity calculation.
        model_name (str): Pretrained SentenceTransformer model name from hugging face
        random_state (int): Seed for reproducibility.

    Returns:
        float: Baseline similarity score (average similarity for random pairs).
    """
    # Ensure reproducibility
    np.random.seed(random_state)

    # Extract unique sentences from the specified column
    texts = text_corpus.dropna().unique()

    # Check if there are enough sentences for sampling
    if len(texts) < 2:
        raise ValueError("Not enough unique texts in the column to calculate similarity.")
    model = SentenceTransformer(model_name)
    if sample_size >=len(text_corpus):
        embeddings = model.encode(text_corpus)
        similarities = model.similarity(embeddings, embeddings)
        return similarities.triu(1).mean()
    else:
        mean = []
        similarities = None
        for i in range(5):
            sample = text_corpus.sample(n=sample_size, random_state=42).tolist()
            embeddings = model.encode(sample)
            similarities = model.similarity(embeddings, embeddings)
            mean.append(similarities.triu(1).mean())
        return np.mean(mean)

class Evaluator:
    def __init__(self, sentence_transformer_name="NeuML/pubmedbert-base-embeddings", ner_model_name="blaze999/Medical-NER"):
        self.transformer = SentenceTransformer(sentence_transformer_name)
        self.ner_pipeline = pipeline("token-classification", model=ner_model_name)

    def compute_cosine_similarity(self, text1:str, text2:str) -> float:
        embeddings = self.transformer.encode([text1, text2])
        similarities = self.transformer.similarity(embeddings[0], embeddings[1]) # cosine similarity
        return similarities[0][0].item()

    def compute_ner_overlap(self, text1, text2):
        ner_results_text1 = self.ner_pipeline(text1)
        ner_results_text2 = self.ner_pipeline(text2)
        entities_text1 = {entity["word"] for entity in ner_results_text1}
        entities_text2 = {entity["word"] for entity in ner_results_text2}
        intersection = entities_text1.intersection(entities_text2)
        union = entities_text1.union(entities_text2)
        return len(intersection) / len(union) if union else 0.0

    def compute_weighted_similarity(self, text1, text2, weight_ner=0.5):
        sentence_similarity = self.compute_cosine_similarity(text1, text2)
        ner_similarity = self.compute_ner_overlap(text1, text2)
        return (1 - weight_ner) * sentence_similarity + weight_ner * ner_similarity

    def compute_dataframe_similarity(self, df, col1_name, col2_name, col_prefixes = ""):
        df[col_prefixes + "ner_score"] = df.apply(lambda x: self.compute_ner_overlap(x[col1_name], x[col2_name]), axis=1)
        df[col_prefixes + "embedding_score"] = df.apply(lambda x: self.compute_cosine_similarity(x[col1_name], x[col2_name]), axis=1)
        df[col_prefixes + "weighted_score"] = df.apply(lambda x: x[col_prefixes + "embedding_score"] * x[col_prefixes + "ner_score"], axis=1)
        return df