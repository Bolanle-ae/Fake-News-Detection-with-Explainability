# 📰 Fake News Detection with Explainable AI (BERT + LIME)

This project uses **Natural Language Processing (NLP)** and **Explainable AI (XAI)** to detect fake news articles and explain the model’s predictions.  
It leverages **BERT**, a state-of-the-art transformer model for text classification, along with **LIME** (Local Interpretable Model-agnostic Explanations) for interpretability.

---

## 🚀 Overview
The model classifies news articles as either **Fake** or **Real** using a fine-tuned **BERT-base-uncased** model.  
Additionally, **LIME** is used to visualize and interpret the most influential words contributing to the classification decision, making the model transparent and explainable.

---

## 🧩 Features
- 🧠 **Fake vs Real News Detection** using BERT
- 🧹 **Text Cleaning and Preprocessing**
- 🔤 **Transformers-based Sequence Classification**
- 🔍 **Explainability with LIME (feature importance visualization)**
- 💡 **End-to-end pipeline** from dataset loading to explainable prediction

---

## 🛠 Technologies Used
### **Programming & Frameworks**
- Python 3.x
- PyTorch
- Transformers (Hugging Face)
- scikit-learn
- LIME (Local Interpretable Model-Agnostic Explanations)

### **Libraries**
- `pandas`, `numpy`, `re`, `string`
- `transformers`
- `torch`
- `lime`
- `scikit-learn`

---

## 📂 Dataset
The project uses two CSV datasets:
- **Fake.csv** → Contains fake news articles  
- **True.csv** → Contains real news articles  

Each file includes:
- `title` — Headline of the article  
- `text` — Article content  
- `subject` — Topic or category of the article  
- `date` — Publication date  

After loading, a binary label is added:
- `0` → Fake News  
- `1` → Real News  

---

## ⚙️ How It Works
1. **Load Datasets**  
   Combines the `Fake.csv` and `True.csv` datasets with labels.

2. **Text Cleaning**  
   Removes punctuation, links, numbers, and extra whitespace; converts text to lowercase.

3. **Data Splitting**  
   Uses `train_test_split()` to create training and testing sets.

4. **Model Initialization**  
   Loads the pretrained **BERT-base-uncased** model for sequence classification.

5. **Prediction**  
   The model predicts whether a news article is *Fake* or *Real*.

6. **Explainability with LIME**  
   LIME highlights the most influential words that led to the model’s decision, increasing interpretability.

---

## 🧠 Example Output

