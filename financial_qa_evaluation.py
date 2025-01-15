import numpy as np
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from bert_score import score
import nltk
nltk.download('punkt')

# Financial QA Dataset
financial_qa_dataset = [
    {
        "question": "What is compound interest?",
        "reference_answer": "Compound interest is interest calculated on both the initial principal and accumulated interest from previous periods, leading to exponential growth of the investment over time.",
        "generated_answer": "Compound interest occurs when interest is added to the principal sum, and from that moment on, the added interest also earns interest."
    },
    {
        "question": "What is diversification in investment?",
        "reference_answer": "Diversification is a risk management strategy that involves spreading investments across different assets, sectors, or markets to reduce exposure to any single investment risk.",
        "generated_answer": "Diversification means spreading your investments across various assets to minimize risk and protect against market volatility."
    },
    {
        "question": "What is a bull market?",
        "reference_answer": "A bull market is a period of time in financial markets when the price of assets is rising or expected to rise, characterized by investor optimism and confidence.",
        "generated_answer": "A bull market represents a period when stock prices are rising and market confidence is high."
    },
    {
        "question": "What is a P/E ratio?",
        "reference_answer": "The Price-to-Earnings (P/E) ratio is a valuation metric that compares a company's stock price to its earnings per share, helping investors assess if a stock is overvalued or undervalued.",
        "generated_answer": "P/E ratio is a metric that shows how much investors are willing to pay for each dollar of a company's earnings."
    },
    {
        "question": "What is a mutual fund?",
        "reference_answer": "A mutual fund is a professionally managed investment vehicle that pools money from multiple investors to purchase a diversified portfolio of stocks, bonds, or other securities.",
        "generated_answer": "A mutual fund pools money from many investors to invest in a diverse portfolio of stocks, bonds, and other securities."
    },
    {
        "question": "What is inflation?",
        "reference_answer": "Inflation is the rate at which the general level of prices for goods and services is rising, consequently decreasing the purchasing power of currency over time.",
        "generated_answer": "Inflation refers to the general increase in prices of goods and services, which reduces the purchasing power of money."
    },
    {
        "question": "What is a bond yield?",
        "reference_answer": "Bond yield is the return an investor realizes on a bond, calculated as the bond's coupon rate divided by its current market price.",
        "generated_answer": "Bond yield represents the return on investment for a bond, expressed as a percentage of the bond's price."
    },
    {
        "question": "What is market capitalization?",
        "reference_answer": "Market capitalization is the total value of a company's outstanding shares, calculated by multiplying the current stock price by the number of shares outstanding.",
        "generated_answer": "Market capitalization is the total dollar value of a company's shares, calculated by multiplying share price by total shares outstanding."
    },
    {
        "question": "What is a dividend?",
        "reference_answer": "A dividend is a distribution of a company's earnings to shareholders, usually paid in cash on a quarterly basis as a reward for investing in the company.",
        "generated_answer": "A dividend is a payment made by a corporation to its shareholders, usually as a distribution of profits."
    },
    {
        "question": "What is a bear market?",
        "reference_answer": "A bear market is a condition in which securities prices fall 20% or more from recent highs amid widespread pessimism and negative investor sentiment.",
        "generated_answer": "A bear market occurs when stock prices fall significantly, typically defined as a drop of 20% or more from recent highs."
    },
    {
        "question": "What is an ETF?",
        "reference_answer": "An Exchange-Traded Fund (ETF) is a type of investment fund traded on stock exchanges, holding assets such as stocks, bonds, or commodities that typically tracks an underlying index.",
        "generated_answer": "An ETF is a basket of securities that trades on an exchange like a stock and typically tracks an underlying index."
    },
    {
        "question": "What is asset allocation?",
        "reference_answer": "Asset allocation is an investment strategy that aims to balance risk and reward by distributing portfolio assets according to an individual's goals, risk tolerance, and investment horizon.",
        "generated_answer": "Asset allocation involves dividing investments among different asset categories like stocks, bonds, and cash based on risk tolerance and goals."
    },
    {
        "question": "What is a margin call?",
        "reference_answer": "A margin call occurs when a broker demands that an investor deposit additional money or securities to cover possible losses in a margin account that has fallen below the required amount.",
        "generated_answer": "A margin call is when brokers require investors to add funds to their margin account because the account value has fallen below the minimum requirement."
    },
    {
        "question": "What is liquidity?",
        "reference_answer": "Liquidity refers to the degree to which an asset can be quickly bought or sold in the market without affecting its price, with cash being the most liquid asset.",
        "generated_answer": "Liquidity is how easily an asset can be converted to cash without significant loss in value."
    },
    {
        "question": "What is a hedge fund?",
        "reference_answer": "A hedge fund is an alternative investment vehicle that pools capital from accredited investors and invests in a variety of assets, often using complex strategies and leverage.",
        "generated_answer": "A hedge fund is a private investment partnership that uses various strategies to generate returns, typically catering to high-net-worth investors."
    },
    {
        "question": "What is short selling?",
        "reference_answer": "Short selling is an investment strategy where an investor borrows shares and immediately sells them, hoping to buy them back later at a lower price, profit from the difference.",
        "generated_answer": "Short selling involves selling borrowed shares with the intention of buying them back at a lower price to profit from the price decline."
    },
    {
        "question": "What is a derivative?",
        "reference_answer": "A derivative is a financial contract whose value is derived from the performance of an underlying asset, index, or entity, used for hedging risks or speculation.",
        "generated_answer": "A derivative is a financial instrument whose value depends on the performance of another underlying asset or benchmark."
    },
    {
        "question": "What is forex trading?",
        "reference_answer": "Forex trading is the buying and selling of currencies on the foreign exchange market, with the aim of profiting from changes in exchange rates between currency pairs.",
        "generated_answer": "Forex trading involves exchanging one currency for another in the foreign exchange market to profit from currency value fluctuations."
    },
    {
        "question": "What is a credit score?",
        "reference_answer": "A credit score is a numerical expression based on a statistical analysis of a person's credit files, representing their creditworthiness and likelihood of repaying debts.",
        "generated_answer": "A credit score is a number that represents an individual's creditworthiness, based on their credit history and financial behavior."
    },
    {
        "question": "What is an IPO?",
        "reference_answer": "An Initial Public Offering (IPO) is the process of offering shares of a private corporation to the public for the first time, allowing companies to raise capital from public investors.",
        "generated_answer": "An IPO is when a private company first sells shares to the public, becoming a publicly traded company."
    }
]

def calculate_bleu_scores():
    smoothie = SmoothingFunction().method1
    bleu_scores = []
    
    for qa_pair in financial_qa_dataset:
        reference = [qa_pair['reference_answer'].split()]
        candidate = qa_pair['generated_answer'].split()
        
        bleu_score = sentence_bleu(reference, candidate, smoothing_function=smoothie)
        bleu_scores.append(bleu_score)
    
    return np.mean(bleu_scores)

def calculate_rouge_scores():
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    rouge_scores = []
    
    for qa_pair in financial_qa_dataset:
        scores = scorer.score(qa_pair['reference_answer'], qa_pair['generated_answer'])
        rouge_scores.append(scores)
    
    # Calculate average scores
    avg_rouge1 = np.mean([score['rouge1'].fmeasure for score in rouge_scores])
    avg_rouge2 = np.mean([score['rouge2'].fmeasure for score in rouge_scores])
    avg_rougeL = np.mean([score['rougeL'].fmeasure for score in rouge_scores])
    
    return {
        'rouge1': avg_rouge1,
        'rouge2': avg_rouge2,
        'rougeL': avg_rougeL
    }

def calculate_bert_scores():
    references = [qa_pair['reference_answer'] for qa_pair in financial_qa_dataset]
    candidates = [qa_pair['generated_answer'] for qa_pair in financial_qa_dataset]
    
    P, R, F1 = score(candidates, references, lang='en', verbose=True)
    
    return {
        'precision': P.mean().item(),
        'recall': R.mean().item(),
        'f1': F1.mean().item()
    }

def main():
    # Calculate BLEU scores
    bleu_score = calculate_bleu_scores()
    print(f"\nAverage BLEU Score: {bleu_score:.4f}")
    
    # Calculate ROUGE scores
    rouge_scores = calculate_rouge_scores()
    print("\nROUGE Scores:")
    print(f"ROUGE-1: {rouge_scores['rouge1']:.4f}")
    print(f"ROUGE-2: {rouge_scores['rouge2']:.4f}")
    print(f"ROUGE-L: {rouge_scores['rougeL']:.4f}")
    
    # Calculate BERTScore
    bert_scores = calculate_bert_scores()
    print("\nBERTScore:")
    print(f"Precision: {bert_scores['precision']:.4f}")
    print(f"Recall: {bert_scores['recall']:.4f}")
    print(f"F1: {bert_scores['f1']:.4f}")

if __name__ == "__main__":
    main()
