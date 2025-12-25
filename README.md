📧 Spam Email Classification using Machine Learning
📌 Project Overview

This project is a machine learning pipeline for spam email detection.
It performs data preprocessing, text cleaning, visualization, feature extraction, model training, evaluation, comparison, ROC analysis, and cross-validation.

The goal is to compare multiple classification models and determine the best one for spam detection based on metrics such as Accuracy, Precision, Recall, F1-score, and ROC-AUC.

🧠 Models Used

The following models are implemented and compared:

Naive Bayes (MultinomialNB)

Logistic Regression

Decision Tree

Random Forest

📂 Dataset Requirements

The dataset should be a CSV file containing:

A text column (e.g. message, text, email)

A label column (e.g. spam, label, category)

Supported label formats:

ham / spam

0 / 1

"0" / "1"

Example:

message,spam
"Congratulations! You won a prize",1
"Meeting tomorrow at 10am",0

⚙️ Project Structure

The code is organized into logical blocks:

Library Imports & Settings

Data Loading

Label Processing

Text Preparation

Text Preprocessing (cleaning & lemmatization)

Exploratory Data Analysis & Visualization

Word Cloud Analysis

Train-Test Split

Text Vectorization

Naive Bayes Model

Logistic Regression (GridSearchCV)

Decision Tree (GridSearchCV)

Random Forest (GridSearchCV)

Model Comparison

Confusion Matrices

ROC Curves

Cross-Validation

Final Report & Recommendations

🧹 Text Preprocessing

The text cleaning pipeline includes:

Lowercasing

Removing emails, URLs, HTML tags

Removing punctuation and digits

Stopword removal

Lemmatization

Token filtering

🔡 Feature Extraction

Two vectorization techniques are used:

TF-IDF (for Logistic Regression, Decision Tree, Random Forest)

Bag of Words (CountVectorizer) (for Naive Bayes)

📊 Evaluation Metrics

Each model is evaluated using:

Accuracy

Precision

Recall

F1-score

ROC-AUC

Confusion Matrix

ROC Curve

📈 Visualization

The project generates:

Class distribution plots

Text length distributions

Average word counts

Word clouds for spam messages

Confusion matrices

ROC curves for all models

Model comparison bar charts

🔁 Cross-Validation

5-Fold Cross-Validation

Accuracy-based evaluation

Mean, Standard Deviation, Min & Max accuracy reported

🏆 Model Selection

The best model is selected based on:

Highest F1-score

Overall performance consistency

ROC-AUC score

🛠️ Libraries Used

pandas

numpy

matplotlib

seaborn

scikit-learn

nltk

wordcloud

🚀 How to Run

Install dependencies:

pip install pandas numpy matplotlib seaborn scikit-learn nltk wordcloud


Download NLTK resources (automatically handled in code)

Place your dataset:

df = pd.read_csv("/content/emails.csv")


Run the notebook or script.

📌 Conclusion

This project demonstrates a complete NLP + Machine Learning workflow for spam detection, from raw data to model evaluation and comparison.
It is suitable for:

Academic coursework

NLP practice

Binary text classification tasks
