# 📧 Spam Email Classification using Machine Learning

![Python](https://img.shields.io/badge/Python-3.x-blue.svg)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-NLP-green.svg)
![Status](https://img.shields.io/badge/Status-Completed-success.svg)

---

## 📌 Project Overview

This project implements a **complete Machine Learning & NLP pipeline** for **spam email classification**.
It covers all stages of a real-world ML workflow — from **raw text preprocessing** to **model evaluation, comparison, and selection**.

The main objective is to **compare multiple classification algorithms** and determine the **best-performing model** using industry-standard metrics.

---

## 🧠 Models Implemented

The following classification models are trained and evaluated:

* 🟢 **Naive Bayes**
* 🔵 **Logistic Regression**
* 🟠 **Decision Tree Classifier**
* 🔴 **Random Forest Classifier**

---

## 📂 Dataset Requirements

The dataset must be provided as a **CSV file** with:

* **Text column** (e.g. `message`, `text`, `email`)
* **Label column** (e.g. `spam`, `label`, `category`)

### ✅ Supported Label Formats

| Format  | Example      |
| ------- | ------------ |
| Text    | `ham / spam` |
| Integer | `0 / 1`      |
| String  | `"0" / "1"`  |

### 📄 Example Dataset

```csv
message,spam
"Congratulations! You won a prize",1
"Meeting tomorrow at 10am",0
```

---

## ⚙️ Project Workflow

The project is structured into clearly defined stages:

1. 📦 Library Imports & Settings
2. 📥 Data Loading
3. 🏷️ Label Processing
4. 🧹 Text Preparation
5. ✂️ Text Preprocessing & Lemmatization
6. 📊 Exploratory Data Analysis (EDA)
7. ☁️ Word Cloud Analysis
8. 🔀 Train-Test Split
9. 🔡 Text Vectorization
10. 🤖 Model Training
11. 📈 Model Evaluation
12. 🔁 Cross-Validation
13. 🏆 Model Comparison & Selection
14. 📝 Final Report & Recommendations

---

## 🧹 Text Preprocessing Pipeline

The text cleaning pipeline includes:

* Lowercasing
* Removing email addresses
* Removing URLs
* Removing HTML tags
* Removing punctuation & digits
* Stopword removal
* Lemmatization
* Token filtering

---

## 🔡 Feature Extraction

Two vectorization techniques are applied:

| Model               | Vectorization                      |
| ------------------- | ---------------------------------- |
| Naive Bayes         | **Bag of Words** |
| Logistic Regression | **TF-IDF**                         |
| Decision Tree       | **TF-IDF**                         |
| Random Forest       | **TF-IDF**                         |

---

## 📊 Evaluation Metrics

Each model is evaluated using:

* Accuracy
* Precision
* Recall
* F1-score
* ROC-AUC
* Confusion Matrix
* ROC Curve

---

## 📈 Visualizations Generated

The project automatically generates:

* 📊 Class distribution plots
* 📏 Text length distribution
* 🧮 Average word count analysis
* ☁️ Word clouds for spam emails
* 🧩 Confusion matrices
* 📈 ROC curves for all models
* 🏆 Model comparison bar charts

---

## 🔁 Cross-Validation

* **5-Fold Cross-Validation**
* Accuracy-based evaluation
* Reported statistics:

  * Mean Accuracy
  * Standard Deviation
  * Minimum Accuracy
  * Maximum Accuracy

---

## 🏆 Model Selection Criteria

The best model is selected based on:

* ✅ Highest **F1-score**
* 📈 Strong **ROC-AUC**
* 🔄 Consistent performance across folds

---

## 🛠️ Libraries Used

```bash
pandas
numpy
matplotlib
seaborn
scikit-learn
nltk
wordcloud
```

---

## 🚀 How to Run the Project

### 1️⃣ Install Dependencies

```bash
pip install pandas numpy matplotlib seaborn scikit-learn nltk wordcloud
```

### 2️⃣ Dataset Placement

Update the dataset path in the code:

```python
df = pd.read_csv("/content/emails.csv")
```

### 3️⃣ Run the Notebook / Script

NLTK resources are downloaded automatically during execution.

---

## 📌 Conclusion

This project demonstrates a **full NLP + Machine Learning workflow** for binary text classification.

### 🎯 Suitable for:

* NLP & ML practice
* Spam detection systems
* Binary text classification problems

---
Kungurtsev Nikita


