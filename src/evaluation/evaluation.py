import pandas as pd
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import numpy as np

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