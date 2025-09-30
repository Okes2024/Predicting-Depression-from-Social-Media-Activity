Predicting Depression from Social Media Activity (Synthetic Data)
📌 Overview

This project explores the use of machine learning to predict the likelihood of depression based on synthetic social media activity data.
The dataset includes behavioral features (posting patterns, sentiment, interaction rates) and text data (synthetic posts), enabling experimentation with NLP and classification models.

⚠️ Disclaimer: This dataset is fully synthetic and not derived from real user data. It is intended for educational and research purposes only.

🗂️ Project Structure
├── predict_depression_from_social_media.py   # Main script: generates dataset, trains models, evaluates
├── synthetic_depression_dataset.xlsx         # Synthetic dataset (>100 samples)
├── outputs/                                  # Saved plots and visualizations
│   ├── roc_curves.png
│   ├── confusion_matrices.png
│   └── text_feature_importance.png
└── README.md                                 # Project documentation

⚙️ Features

Synthetic dataset generation with:

Posting behavior metrics (e.g., posting frequency, night-time activity).

Sentiment-based features (mean sentiment, variance, negative word rate).

NLP text data (synthetic user posts with positive/negative/neutral words).

ML pipelines for:

Logistic Regression

Random Forest Classifier

Evaluation metrics:

Accuracy, ROC AUC, Confusion Matrix, Classification Report

ROC Curves & Feature Importances

📊 Example Outputs

ROC Curve: Comparison of Logistic Regression vs Random Forest

Confusion Matrices: Classification performance visualization

Top Text Features: Indicative n-grams for depression prediction

🚀 Getting Started
1️⃣ Install Dependencies
pip install numpy pandas scikit-learn matplotlib seaborn

2️⃣ Run the Script
python predict_depression_from_social_media.py

3️⃣ Dataset

A synthetic dataset (synthetic_depression_dataset.xlsx) will be generated automatically.

🧪 Usage

Experiment with preprocessing (scaling, TF-IDF).

Extend with deep learning models (LSTM, BERT).

Replace synthetic dataset with real-world anonymized datasets (with ethical approval).

📌 Limitations

Dataset is synthetic → may not capture real-world complexity.

Models trained here are not suitable for clinical use.

Intended purely for educational and research experimentation.

👨‍💻 Author

Imoni Okes
