# Text Summarization System 📝

This repository contains the implementation of **Project 4** for the **Natural Language Processing (NLP)** course at the Faculty of Computing & Artificial Intelligence, Capital University (formerly Helwan University).

---

## 🏫 Institutional Information
- **University:** Capital University [formerly Helwan University]
- **Faculty:** Computing & Artificial Intelligence
- **Course:** Natural Language Processing (NLP)
- **Semester:** Spring "Semester 2" 2025-2026

---

## 1. Objective
The primary goal of this project is to build a system that summarizes long text into a shorter version while:
* **Preserving Meaning:** Ensuring the main ideas are not lost.
* **Information Density:** Reducing length while keeping the most important information.
* **Quality:** Generating coherent summaries rather than random sentence fragments.

**Example:**
- **Input:** "The food was amazing and the service was excellent. The restaurant was clean and the staff were friendly. However, the delivery was very slow and the order arrived late."
- **Output:** "The food and service were good, but the delivery was slow."

---

## 2. Dataset
- **Source:** https://www.kaggle.com/datasets/gowrishankarp/newspaper-text-summarization-cnn-dailymail ,https://www.tensorflow.org/datasets/catalog/gigaword

---

## 3. System Tasks

### 🛠️ Preprocessing
To prepare the raw text for analysis, the system performs:
1.  **Lowercase Conversion:** Standardizing text to avoid case sensitivity.
2.  **Punctuation Removal:** Eliminating noise from the data.
3.  **Stopwords Removal:** Filtering out common words (e.g., "is", "the") using NLTK/SpaCy.
4.  **Tokenization:** Splitting text into sentences and individual word tokens.

### 📊 Feature Extraction
We represent text using two distinct methods:
-   **TF-IDF:** Statistical weighting of word importance based on frequency.
-   **Word Embeddings:** Capturing semantic meaning using models like Word2Vec, GloVe, or BERT embeddings.

### 🚀 Implementation Methods
1.  **Baseline Method (Extractive):** Uses TF-IDF scores to rank and select the top-scoring sentences that represent the core of the document.
2.  **Advanced Method (Transformer-based):** Utilizes pre-trained models (e.g., BERT or BART) to generate/extract summaries based on deep contextual understanding.

---

## 4. Evaluation
The performance of the system is evaluated through:
* **Length Comparison:** Ensuring the summary is significantly shorter than the original.
* **Information Retention:** Verifying that key keywords and main ideas are present.
* **Method Comparison:** Analyzing the quality difference between the TF-IDF Baseline and the Transformer-based Advanced method.
* **Manual Evaluation:** Qualitative assessment of the summary's readability and flow.
