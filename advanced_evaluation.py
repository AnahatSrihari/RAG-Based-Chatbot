import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict
import re
from financial_qa_dataset import financial_qa_dataset

class AdvancedEvaluator:
    def __init__(self):
        # Load models
        self.tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
        self.model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')

    def get_embeddings(self, text: str) -> np.ndarray:
        """Generate embeddings for input text."""
        inputs = self.tokenizer(text, return_tensors='pt', padding=True, truncation=True)
        with torch.no_grad():
            outputs = self.model(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1).numpy()
        return embeddings

    def calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate cosine similarity between two texts."""
        emb1 = self.get_embeddings(text1)
        emb2 = self.get_embeddings(text2)
        return cosine_similarity(emb1, emb2)[0][0]

    def get_tokens(self, text: str) -> set:
        """Simple tokenization using regex."""
        return set(re.findall(r'\b\w+\b', text.lower()))

    def evaluate_context_relevance(self, question: str, context: str) -> Dict:
        """
        Evaluate how relevant the context is to the question.
        """
        similarity = self.calculate_similarity(question, context)
        
        # Analyze token overlap
        question_tokens = self.get_tokens(question)
        context_tokens = self.get_tokens(context)
        
        token_overlap = len(question_tokens.intersection(context_tokens)) / len(question_tokens) if question_tokens else 0
        
        return {
            'semantic_similarity': float(similarity),
            'token_overlap': token_overlap,
            'overall_relevance': (float(similarity) + token_overlap) / 2
        }

    def evaluate_answer_faithfulness(self, context: str, answer: str) -> Dict:
        """
        Evaluate how faithful the answer is to the context.
        """
        # Calculate overall semantic similarity
        similarity = self.calculate_similarity(context, answer)
        
        # Token-based evaluation
        context_tokens = self.get_tokens(context)
        answer_tokens = self.get_tokens(answer)
        
        token_precision = len(answer_tokens.intersection(context_tokens)) / len(answer_tokens) if answer_tokens else 0
        
        return {
            'semantic_similarity': similarity,
            'token_precision': token_precision,
            'overall_faithfulness': (similarity + token_precision) / 2
        }

    def evaluate_answer_relevance(self, question: str, answer: str) -> Dict:
        """
        Evaluate how relevant the answer is to the question.
        """
        # Direct semantic similarity
        semantic_sim = self.calculate_similarity(question, answer)
        
        # Answer completeness check
        answer_tokens = self.get_tokens(answer)
        completeness = 1.0 if len(answer_tokens) >= 5 else len(answer_tokens) / 5  # Simple heuristic
        
        # Question-answer token alignment
        question_tokens = self.get_tokens(question)
        token_alignment = len(question_tokens.intersection(answer_tokens)) / len(question_tokens) if question_tokens else 0
        
        return {
            'semantic_relevance': float(semantic_sim),
            'token_alignment': token_alignment,
            'answer_completeness': completeness,
            'overall_relevance': (float(semantic_sim) + token_alignment + completeness) / 3
        }

def main():
    evaluator = AdvancedEvaluator()
    
    print("Starting Advanced Evaluation...\n")
    
    total_metrics = {
        'context_relevance': [],
        'answer_faithfulness': [],
        'answer_relevance': []
    }
    
    for qa_pair in financial_qa_dataset:
        question = qa_pair['question']
        reference = qa_pair['reference_answer']
        generated = qa_pair['generated_answer']
        
        print(f"\nEvaluating Question: {question}")
        
        # Context Relevance (using reference answer as context)
        context_metrics = evaluator.evaluate_context_relevance(question, reference)
        print("\nContext Relevance Metrics:")
        print(f"Semantic Similarity: {context_metrics['semantic_similarity']:.3f}")
        print(f"Token Overlap: {context_metrics['token_overlap']:.3f}")
        print(f"Overall Relevance: {context_metrics['overall_relevance']:.3f}")
        
        # Answer Faithfulness
        faithfulness_metrics = evaluator.evaluate_answer_faithfulness(reference, generated)
        print("\nAnswer Faithfulness Metrics:")
        print(f"Semantic Similarity: {faithfulness_metrics['semantic_similarity']:.3f}")
        print(f"Token Precision: {faithfulness_metrics['token_precision']:.3f}")
        print(f"Overall Faithfulness: {faithfulness_metrics['overall_faithfulness']:.3f}")
        
        # Answer Relevance
        relevance_metrics = evaluator.evaluate_answer_relevance(question, generated)
        print("\nAnswer Relevance Metrics:")
        print(f"Semantic Relevance: {relevance_metrics['semantic_relevance']:.3f}")
        print(f"Token Alignment: {relevance_metrics['token_alignment']:.3f}")
        print(f"Answer Completeness: {relevance_metrics['answer_completeness']:.3f}")
        print(f"Overall Relevance: {relevance_metrics['overall_relevance']:.3f}")
        
        # Collect metrics
        total_metrics['context_relevance'].append(context_metrics['overall_relevance'])
        total_metrics['answer_faithfulness'].append(faithfulness_metrics['overall_faithfulness'])
        total_metrics['answer_relevance'].append(relevance_metrics['overall_relevance'])
    
    # Print aggregate results
    print("\n=== Aggregate Results ===")
    print(f"Average Context Relevance: {np.mean(total_metrics['context_relevance']):.3f}")
    print(f"Average Answer Faithfulness: {np.mean(total_metrics['answer_faithfulness']):.3f}")
    print(f"Average Answer Relevance: {np.mean(total_metrics['answer_relevance']):.3f}")

if __name__ == "__main__":
    main()
