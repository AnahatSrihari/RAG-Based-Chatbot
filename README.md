

# **RAG-Based Financial Chatbot**
![image](https://github.com/user-attachments/assets/5ead7884-a0a9-4998-b460-ad92d011e183)


## Overview  
This repository contains the implementation of a **Retrieval-Augmented Generation (RAG)**-based financial chatbot powered by **ChatGPT-3.5 Turbo**. The chatbot is designed to answer user queries based on financial reports (e.g., quarterly, annual reports) with high contextual relevance, accuracy, and actionable insights. It integrates advanced embedding models (**OpenAI ADA** and **MiniLM**) for retrieval and uses **ChatGPT-3.5 Turbo** to generate coherent, grounded responses.

---

## **Features**
- **Dynamic Retrieval**: Retrieves relevant financial data from pre-indexed documents using a vector database (FAISS/Chroma).  
- **Accurate Answers**: Provides factually correct responses, grounded in retrieved financial data.  
- **Advanced Embeddings**: Supports both **OpenAI ADA** and **MiniLM** for semantic data representation.  
- **Real-Time Query Handling**: Efficiently processes queries with token-aware response generation.  
- **Interactive Interface**: User-friendly interaction via **Gradio UI**.  

---

## **Project Structure**
```plaintext
├── data/                   # Financial reports and preprocessed data
├── embeddings/             # Scripts for embedding document chunks
├── retrieval/              # Retrieval system using FAISS/Chroma
├── generation/             # ChatGPT-3.5 Turbo integration
├── ui/                     # Gradio-based user interface
├── evaluation/             # Scripts for BLEU, ROUGE, and BERTScore evaluation
├── utils/                  # Helper functions for preprocessing and orchestration
├── requirements.txt        # Python dependencies
├── README.md               # Project documentation
└── main.py                 # Main script to run the chatbot
```

---

## **Getting Started**

### **1. Clone the Repository**  
```bash
git clone https://github.com/yourusername/RAG-Financial-Chatbot.git
cd RAG-Financial-Chatbot
```

### **2. Set Up Environment**  
Install the required Python dependencies:  
```bash
pip install -r requirements.txt
```

### **3. Download Financial Data**  
Add financial reports (e.g., PDFs of quarterly/annual reports) to the `data/` directory. Use **OCR tools** to preprocess the reports into machine-readable text.

### **4. Configure API Keys**  
Add your **OpenAI API key** and any required embedding model credentials in a `.env` file:  
```plaintext
OPENAI_API_KEY=your_openai_key
```

### **5. Run the Application**  
Start the chatbot with:  
```bash
python main.py
```
Access the chatbot via the **Gradio UI** at the displayed local/online URL.

---

## **Usage**
### Query Examples
1. **Quantitative**:  
   - *"What was the company’s revenue growth last quarter?"*  
   - *"What is the debt-to-equity ratio for the fiscal year?"*  
2. **Qualitative**:  
   - *"What are the key risks highlighted in the Q3 report?"*  
   - *"How has the company’s strategy evolved over the past year?"*

### Output  
The chatbot generates concise, context-aware responses by retrieving relevant financial data and synthesizing it with **ChatGPT-3.5 Turbo**.

---

## **Evaluation**
The chatbot’s performance is evaluated using:  
1. **BLEU**: Measures precision of n-gram overlap between generated and reference answers.  
2. **ROUGE**: Assesses recall for narrative-style responses.  
3. **BERTScore**: Evaluates semantic similarity between generated and reference answers.  

Evaluation scripts are located in the `evaluation/` directory.

---

## **Key Results**
- **OpenAI ADA + ChatGPT-3.5 Turbo**:  
   - High accuracy and semantic relevance for complex queries.  
   - Best suited for detailed financial analysis.  
- **MiniLM + ChatGPT-3.5 Turbo**:  
   - Faster response times with moderate accuracy.  
   - Ideal for performance-critical scenarios.

---

## **Future Enhancements**
- **Integration of GPT-4**: Enhance response accuracy for nuanced queries.  
- **Real-Time Data**: Incorporate APIs for live stock prices and market updates.  
- **Improved Visual Data Handling**: Enable interpretation of tables, charts, and graphs.  
- **Multi-Domain Expansion**: Adapt for legal, medical, or scientific documents.

---

## **Contributing**
Contributions are welcome!  
1. Fork this repository.  
2. Create a new branch:  
   ```bash
   git checkout -b feature/your-feature
   ```  
3. Commit your changes:  
   ```bash
   git commit -m "Add your feature"
   ```  
4. Push to the branch:  
   ```bash
   git push origin feature/your-feature
   ```  
5. Submit a pull request.

---

## **Acknowledgments**
- **OpenAI** for providing the ChatGPT API.  
- **Hugging Face** for embedding model resources.  
- **FAISS** for efficient vector similarity search.  

For any questions or issues, feel free to open an issue or contact us 

--- 

