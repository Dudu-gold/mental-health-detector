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
| random forest | 74% | 0.74 |
| Support Vector Machine (LinearSVC) | 77% | 0.77 |

## Why Lightweight Models?
This project deliberately employs lightweight models — random forest and Support Vector Machine — instead of computationally expensive transformer based models like BERT. This makes the solution more accessible, interpretable, and deployable in resource constrained environments.

## Key Findings
- Random Forest achieved 74% accuracy while LinearSVC (SVM) achieved 77% accuracy
- Normal class achieved the highest F1-score of 0.90 due to having the most training samples (18,391 posts)
- Suicidal class achieved the lowest F1-score of 0.64 due to overlapping vocabulary with the Depression category
- Dataset imbalance significantly affected per-class performance — particularly for the Anxiety class (5,503 posts)
- Cultural language differences and informal expressions were identified as key limitations
- A real-world test using Nigerian Pidgin ("I don dey tire for everything. E be like say nothing dey work for my life. I no fit carry this thing again.") returned a Suicidal prediction with only 37% confidence — barely above a random guess across four classes — confirming that the model cannot reliably detect mental health risk expressed outside Western English, and highlighting the urgent need for culturally diverse and multilingual training datasets

## Project Structure
```
mental_health_project/
│
├── model/
│   ├── rf_model.pkl
│   ├── svm_model.pkl
│   └── vectorizer.pkl
│
├── notebooks/
│   ├── exploration.ipynb
│   ├── text_processing.ipynb
│   └── tfidf_model.ipynb
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
| Streamlit | Web application  |

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
Final Year Project — 2026

## License
MIT License — see LICENSE file for details
```


