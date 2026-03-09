# 🛍 Fine-Grained Sentiment Analysis of Amazon Product Reviews using Transformer Models

## 📌 Project Overview

This project implements a **fine-grained sentiment analysis system** for Amazon product reviews using **BERT (Bidirectional Encoder Representations from Transformers)**.

Unlike traditional sentiment analysis that predicts only **Positive, Neutral, or Negative**, this system performs **5-class sentiment classification** to capture more nuanced opinions expressed in product reviews.

The model is fine-tuned using a large Amazon review dataset and deployed as an interactive **Streamlit web application** that allows users to analyze reviews in real time.

---

## 🎯 Objectives

The main objectives of this project are:

- Build a **Transformer-based NLP model** for sentiment classification.
- Perform **fine-grained sentiment analysis** with 5 sentiment categories.
- Train and evaluate a **BERT model** on Amazon product reviews.
- Develop a **web application** to interactively analyze review sentiments.
- Visualize sentiment distributions using charts.

---

## 🧠 Sentiment Classes

The model predicts **five sentiment classes**:

| Label | Sentiment |
|------|-----------|
| 0 | Very Negative |
| 1 | Negative |
| 2 | Neutral |
| 3 | Positive |
| 4 | Very Positive |

This approach allows the model to distinguish between **mild and strong opinions**, making it more informative than simple 3-class sentiment models.

---

## 📂 Dataset

The model is trained on the **Amazon Reviews Fine-Grained Sentiment Dataset**.

Dataset Link:

https://www.kaggle.com/datasets/yacharki/amazonreviews-for-sentianalysis-finegrained-csv

Dataset files used:

```
train.csv
test.csv
```

Dataset columns used in the project:

| Column | Description |
|------|-------------|
| class_index | Star rating (1–5) |
| review_text | Full customer review text |

The star ratings are converted into sentiment labels using:

```
label = class_index - 1
```

This produces labels from **0 to 4** for model training.

> The dataset is **not included in this repository** due to GitHub file size limitations.

---

## ⚙️ Data Preprocessing

The following preprocessing steps are applied:

- Convert text to lowercase
- Remove URLs
- Normalize whitespace
- Tokenize text using **BERT tokenizer**
- Convert labels to numerical sentiment classes

---

## 🤖 Model Architecture

The project uses the **BERT Transformer model** from Hugging Face.

Model used:

```
bert-base-uncased
```

Architecture pipeline:

```
Input Review Text
        ↓
BERT Tokenizer
        ↓
BERT Transformer Encoder
        ↓
Classification Layer (5 outputs)
        ↓
Softmax Probability Distribution
```

The model outputs the probability of each sentiment class.

---

## 🏋️ Model Training

The model is trained using **PyTorch** and the **Hugging Face Transformers library**.

Training configuration:

| Parameter | Value |
|--------|------|
| Model | BERT Base |
| Epochs | 2 |
| Batch Size | 16 |
| Learning Rate | 5e-5 |
| Loss Function | CrossEntropyLoss |
| Optimizer | AdamW |

Training was performed using **fine-tuning of a pretrained BERT model**.

---

## 📊 Model Evaluation

The model is evaluated using:

- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix

Example evaluation output:

```
Test Accuracy ≈ 0.53
```

This accuracy is expected because **5-class sentiment classification is significantly harder than 3-class sentiment analysis**.

---

## 🌐 Streamlit Web Application

A **Streamlit web interface** was developed to make the model interactive.

The application allows users to:

### 🔎 Analyze a Single Review

Users can input any review text and the model predicts its sentiment.

The interface displays:

- Predicted sentiment
- Confidence scores
- Sentiment probability chart

---

### 📂 Batch Sentiment Analysis

Users can upload a CSV file containing reviews.

The app will:

- Predict sentiment for each review
- Display results in a table
- Generate sentiment distribution charts

Visualizations include:

- 📊 Bar Chart
- 🥧 Pie Chart

---

## 🗂 Project Structure

```
amazon-review-sentiment-transformer/
│
├── src/
│   ├── train.py
│   ├── evaluate.py
│   ├── predict.py
│   └── data_preprocessing.py
│
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── models/      (ignored in GitHub)
├── data/        (ignored in GitHub)
```

---

## 🚀 Installation

Clone the repository:

```
git clone https://github.com/yourusername/amazon-review-sentiment-transformer.git
cd amazon-review-sentiment-transformer
```

Create virtual environment:

```
python -m venv venv
source venv/bin/activate
```

Install dependencies:

```
pip install -r requirements.txt
```

---

## 🏃 Running the Project

### Train the Model

```
python src/train.py
```

---

### Evaluate the Model

```
python src/evaluate.py
```

---

### Test Custom Reviews

```
python src/predict.py
```

---

### Launch the Web App

```
streamlit run app.py
```

The application will open in your browser.

---

## 📈 Future Improvements

Possible enhancements include:

- Training on the **full dataset for higher accuracy**
- Using **larger transformer models such as RoBERTa or DeBERTa**
- Implementing **aspect-based sentiment analysis**
- Deploying the web app on **Streamlit Cloud or HuggingFace Spaces**

---

## 👨‍💻 Authors

- Chaitanya Reddy  
- Shubham Dansena  
- K C Karthik  

---

