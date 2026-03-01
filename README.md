# Mental Health Risk Detection from Social Media Text using Lightweight Models: A Machine Learning Approach with Bias and Cultural Analysis

## Overview
This project develops a machine learning system that automatically detects mental health risk from social media text. Unlike studies that focus solely on prediction accuracy, this project goes further by analyzing model performance disparities caused by dataset imbalance, cultural language differences, and variation in expression across mental health categories.

## Problem Statement
Mental health conditions are increasingly expressed through social media platforms. Early detection of such expressions can facilitate timely intervention. However, most existing models fail to address performance bias across different demographic and cultural groups — a gap this project directly investigates.

## Mental Health Categories
- 🟢 Normal
- 🟡 Depression
- 🟠 Anxiety
- 🔴 Suicidal

## Models Used
| Model | Accuracy | Weighted F1 |
|-------|---------|-------------|
| Logistic Regression | 77% | 0.77 |
| Support Vector Machine (SVM) | 77% | 0.76 |

## Why Lightweight Models?
This project deliberately employs lightweight models — Logistic Regression and Support Vector Machine — instead of computationally expensive transformer based models like BERT. This makes the solution more accessible, interpretable, and deployable in resource constrained environments.

## Key Findings
- Both models achieved 77% overall accuracy
- Normal class achieved highest F1 score of 0.90 due to having the most training samples
- Suicidal class achieved lowest F1 score of 0.64 due to overlapping vocabulary with Depression
- Dataset imbalance significantly affected per class performance
- Cultural language differences and informal expressions were identified as limitations through real world testing

## Project Structure
```
mental_health_project/
│
├── model/
│   ├── lr_model.pkl
│   ├── svm_model.pkl
│   └── vectorizer.pkl
│
├── notebooks/
│   ├── exploration.ipynb
│   ├── text_processing.ipynb
│   └── tfidf.ipynb
│
├── .gitignore
├── LICENSE
├── README.md
└── requirements.txt
```

## Tech Stack
| Tool | Purpose |
|------|---------|
| Python | Programming language |
| Scikit-learn | Model training and evaluation |
| NLTK | Text preprocessing |
| TF-IDF | Feature extraction |
| Matplotlib/Seaborn | Visualization |
| Streamlit | Web application (coming soon) |

## Installation
```bash
git clone https://github.com/Dudu-gold/mental-health-detector.git
cd mental-health-detector
pip install -r requirements.txt
```

## Dataset
Downloaded from Kaggle — Mental Health Social Media Dataset.
Contains 49,000+ social media posts labeled across 4 mental health categories.

## Limitations
- Dataset is predominantly standard English — cultural expressions and slang may be misclassified
- Class imbalance affects model performance across categories
- Lightweight models may not capture deep semantic meaning compared to transformer based models

## Future Work
- Fine-tune pre-trained transformer models such as BERT
- Incorporate diverse cultural and multilingual training data
- Develop real time location based emergency response system
- Address class imbalance using SMOTE or other oversampling techniques

## Author
Duduyemi Olalekan
Final Year Project — 2025

## License
MIT License — see LICENSE file for details
```


