import numpy as np
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from bert_score import score
import nltk
from financial_qa_dataset import financial_qa_dataset

nltk.download('punkt')

def calculate_bleu_scores(dataset):
    smoothie = SmoothingFunction().method1
    bleu_scores = []
    
    for qa_pair in dataset:
        reference = [qa_pair['reference_answer'].split()]
        candidate = qa_pair['generated_answer'].split()
        bleu_score = sentence_bleu(reference, candidate, smoothing_function=smoothie)
        bleu_scores.append(bleu_score)
    
    return np.mean(bleu_scores)

def calculate_rouge_scores(dataset):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    rouge_scores = []
    
    for qa_pair in dataset:
        scores = scorer.score(qa_pair['reference_answer'], qa_pair['generated_answer'])
        rouge_scores.append(scores)
    
    avg_rouge1 = np.mean([score['rouge1'].fmeasure for score in rouge_scores])
    avg_rouge2 = np.mean([score['rouge2'].fmeasure for score in rouge_scores])
    avg_rougeL = np.mean([score['rougeL'].fmeasure for score in rouge_scores])
    
    return {
        'rouge1': avg_rouge1,
        'rouge2': avg_rouge2,
        'rougeL': avg_rougeL
    }

def calculate_bert_scores(dataset):
    references = [qa_pair['reference_answer'] for qa_pair in dataset]
    candidates = [qa_pair['generated_answer'] for qa_pair in dataset]
    
    P, R, F1 = score(candidates, references, lang='en', verbose=True)
    
    return {
        'precision': P.mean().item(),
        'recall': R.mean().item(),
        'f1': F1.mean().item()
    }

def main():
    print("Evaluating Financial QA responses...")
    
    # Calculate BLEU scores
    bleu_score = calculate_bleu_scores(financial_qa_dataset)
    print(f"\nAverage BLEU Score: {bleu_score:.4f}")
    
    # Calculate ROUGE scores
    rouge_scores = calculate_rouge_scores(financial_qa_dataset)
    print("\nROUGE Scores:")
    print(f"ROUGE-1: {rouge_scores['rouge1']:.4f}")
    print(f"ROUGE-2: {rouge_scores['rouge2']:.4f}")
    print(f"ROUGE-L: {rouge_scores['rougeL']:.4f}")
    
    # Calculate BERTScore
    bert_scores = calculate_bert_scores(financial_qa_dataset)
    print("\nBERTScore:")
    print(f"Precision: {bert_scores['precision']:.4f}")
    print(f"Recall: {bert_scores['recall']:.4f}")
    print(f"F1: {bert_scores['f1']:.4f}")

if __name__ == "__main__":
    main()
