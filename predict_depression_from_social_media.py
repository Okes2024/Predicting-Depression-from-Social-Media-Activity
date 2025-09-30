"""
Predicting-Depression-from-Social-Media-Activity (Synthetic Data)
-----------------------------------------------------------------
Generates a synthetic dataset (>100 samples) with text + behavioral features
to predict depression risk from social media activity. Trains ML models and
evaluates with ROC AUC, classification report, and simple interpretability.

Requirements:
    pip install numpy pandas scikit-learn matplotlib seaborn
"""

import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, roc_auc_score, classification_report,
    confusion_matrix, roc_curve
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer

# ---------------------------
# Reproducibility & settings
# ---------------------------
RNG_SEED = 42
random.seed(RNG_SEED)
np.random.seed(RNG_SEED)

N_SAMPLES = 1200                 # >100 as requested
OUTPUT_CSV = "synthetic_social_media_depression.csv"
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

# ---------------------------
# Synthetic text generator
# ---------------------------
NEG_LEX = [
    "sad", "lonely", "tired", "empty", "numb", "hopeless",
    "worthless", "cry", "exhausted", "anxious", "down", "dark",
    "insomnia", "pain", "lost", "overwhelmed", "nobody", "tears"
]
POS_LEX = [
    "happy", "grateful", "excited", "joy", "love", "sunny", "peace",
    "hope", "smile", "strong", "progress", "energy", "friends"
]
NEU_LEX = [
    "work", "project", "coffee", "news", "update", "meeting",
    "study", "music", "weather", "travel", "movie", "dinner"
]

FIRST_PERSON = ["I", "I'm", "I've", "me", "my", "myself"]
EMOJIS_SAD = ["😞", "😔", "💤", "😢", "😩"]
EMOJIS_POS = ["😊", "🌞", "💪", "🎉", "😁"]
EMOJIS_NEU = ["🙂", "📚", "☕", "🎧", "📰"]

def synth_post(neg_weight=0.3, pos_weight=0.3, neu_weight=0.4, emoji_rate=0.2, first_person_rate=0.4):
    n_tokens = np.random.randint(6, 24)
    tokens = []
    for _ in range(n_tokens):
        roll = np.random.rand()
        if roll < neg_weight:
            tok = np.random.choice(NEG_LEX)
        elif roll < neg_weight + pos_weight:
            tok = np.random.choice(POS_LEX)
        else:
            tok = np.random.choice(NEU_LEX)
        if np.random.rand() < first_person_rate:
            tok = np.random.choice(FIRST_PERSON) + " " + tok
        tokens.append(tok)
    # sprinkle emojis
    if np.random.rand() < emoji_rate:
        tokens.append(np.random.choice(EMOJIS_SAD + EMOJIS_POS + EMOJIS_NEU))
    return " ".join(tokens)

# ---------------------------
# Generate synthetic dataset
# ---------------------------
rows = []
for i in range(N_SAMPLES):
    # latent persona
    baseline_mood = np.random.normal(0.0, 1.0)    # lower => worse mood
    night_owl = np.random.rand() < 0.45
    social_support = np.random.beta(2, 2)         # 0..1
    activity_level = np.random.gamma(shape=2.0, scale=7.0)  # posts/week-ish
    activity_level = np.clip(activity_level, 0, 60)

    # features
    avg_daily_posts = max(0, np.random.normal(activity_level/7, 0.8))
    night_posts_ratio = np.clip(np.random.beta(2, 3) + (0.25 if night_owl else 0.0), 0, 1)
    mean_sentiment = np.clip(0.4 + 0.25*baseline_mood + 0.2*(social_support-0.5) + np.random.normal(0, 0.15), 0, 1)
    sentiment_var = np.clip(np.random.gamma(1.2, 0.15), 0, 1)
    interaction_rate = np.clip(np.random.normal(0.4 + 0.5*social_support, 0.15), 0, 1)  # likes+replies normalized
    first_person_rate = np.clip(np.random.normal(0.35 - 0.2*baseline_mood, 0.1), 0, 1)
    negative_word_rate = np.clip(np.random.normal(0.2 - 0.15*baseline_mood, 0.08), 0, 1)
    posting_entropy = np.clip(np.random.normal(0.5 - 0.1*night_posts_ratio, 0.15), 0, 1)  # time-of-day spread
    emoji_rate = np.clip(np.random.beta(2, 6), 0, 1)

    # generate a "profile text" concatenation of recent posts
    neg_w = np.clip(0.25 + 0.35*negative_word_rate, 0.05, 0.9)
    pos_w = np.clip(0.4 + 0.5*(mean_sentiment-0.5), 0.05, 0.9)
    neu_w = max(0.01, 1.0 - neg_w - pos_w)
    docs = [synth_post(neg_w, pos_w, neu_w, emoji_rate=emoji_rate, first_person_rate=first_person_rate) for _ in range(np.random.randint(3, 8))]
    recent_text = " . ".join(docs)

    # latent risk score
    risk = (
        0.45 * (1 - mean_sentiment) +
        0.15 * night_posts_ratio +
        0.12 * negative_word_rate +
        0.10 * (0.5 - interaction_rate) +
        0.08 * first_person_rate +
        0.06 * (0.5 - social_support)
    )
    risk += np.random.normal(0, 0.07)
    risk = np.clip(risk, 0.02, 0.95)
    label = np.random.rand() < risk   # 1 = depression flag

    rows.append({
        "user_id": f"U{i:05d}",
        "avg_daily_posts": avg_daily_posts,
        "night_posts_ratio": night_posts_ratio,
        "mean_sentiment": mean_sentiment,
        "sentiment_var": sentiment_var,
        "interaction_rate": interaction_rate,
        "first_person_rate": first_person_rate,
        "negative_word_rate": negative_word_rate,
        "posting_entropy": posting_entropy,
        "emoji_rate": emoji_rate,
        "recent_text": recent_text,
        "depression_flag": int(label)
    })

df = pd.DataFrame(rows)
csv_path = Path(OUTPUT_CSV).resolve()
df.to_csv(csv_path, index=False)
print(f"Saved synthetic dataset to: {csv_path}")
print("Shape:", df.shape)
print("Class balance:\n", df['depression_flag'].value_counts(normalize=True))

# ---------------------------
# Modeling
# ---------------------------
NUMERIC_FEATS = [
    "avg_daily_posts","night_posts_ratio","mean_sentiment","sentiment_var",
    "interaction_rate","first_person_rate","negative_word_rate",
    "posting_entropy","emoji_rate"
]
TEXT_FEAT = "recent_text"
TARGET = "depression_flag"

X = df[NUMERIC_FEATS + [TEXT_FEAT]]
y = df[TARGET].astype(int)

preprocess = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), NUMERIC_FEATS),
        ("txt", TfidfVectorizer(max_features=3000, ngram_range=(1,2), min_df=2), TEXT_FEAT)
    ]
)

# Models
logreg = Pipeline([
    ("prep", preprocess),
    ("clf", LogisticRegression(max_iter=2000, n_jobs=None, solver="liblinear"))
])

rf = Pipeline([
    ("prep", preprocess),
    ("clf", RandomForestClassifier(n_estimators=300, random_state=RNG_SEED, n_jobs=-1))
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RNG_SEED, stratify=y
)

print("\nTraining Logistic Regression...")
logreg.fit(X_train, y_train)
p_lr = logreg.predict_proba(X_test)[:,1]
y_lr = (p_lr >= 0.5).astype(int)

print("Training Random Forest...")
rf.fit(X_train, y_train)
p_rf = rf.predict_proba(X_test)[:,1]
y_rf = (p_rf >= 0.5).astype(int)

def evaluate(name, y_true, y_pred, y_prob):
    print(f"\n=== {name} ===")
    print("Accuracy:", accuracy_score(y_true, y_pred))
    try:
        print("ROC AUC:", roc_auc_score(y_true, y_prob))
    except Exception as e:
        print("ROC AUC error:", e)
    print(classification_report(y_true, y_pred, target_names=["NoFlag","Flag"]))
    cm = confusion_matrix(y_true, y_pred)
    print("Confusion matrix:\n", cm)
    return cm

cm_lr = evaluate("LogReg", y_test, y_lr, p_lr)
cm_rf = evaluate("RandomForest", y_test, y_rf, p_rf)

# ---------------------------
# Plots
# ---------------------------
plt.figure(figsize=(8,6))
fpr_lr, tpr_lr, _ = roc_curve(y_test, p_lr)
fpr_rf, tpr_rf, _ = roc_curve(y_test, p_rf)
plt.plot(fpr_lr, tpr_lr, label="LogReg")
plt.plot(fpr_rf, tpr_rf, label="RandomForest")
plt.plot([0,1],[0,1],"--", color="gray")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves")
plt.legend()
plt.grid(True)
plt.tight_layout()
roc_path = OUTPUT_DIR/"roc_curves.png"
plt.savefig(roc_path, dpi=160)
plt.close()

fig, axes = plt.subplots(1,2, figsize=(12,5))
sns.heatmap(cm_lr, annot=True, fmt="d", cmap="Blues", cbar=False, ax=axes[0])
axes[0].set_title("Confusion Matrix - LogReg")
axes[0].set_xlabel("Predicted"); axes[0].set_ylabel("Actual")

sns.heatmap(cm_rf, annot=True, fmt="d", cmap="Greens", cbar=False, ax=axes[1])
axes[1].set_title("Confusion Matrix - RandomForest")
axes[1].set_xlabel("Predicted"); axes[1].set_ylabel("Actual")
plt.tight_layout()
cm_path = OUTPUT_DIR/"confusion_matrices.png"
plt.savefig(cm_path, dpi=160)
plt.close()

print(f"Saved plots to: {roc_path} and {cm_path}")

# ---------------------------
# Simple interpretability
# ---------------------------
# Show strongest text features for Logistic Regression (if available)
try:
    vec = logreg.named_steps["prep"].named_transformers_["txt"]
    clf = logreg.named_steps["clf"]
    # ColumnTransformer stacks [num, txt]; get txt feature names
    text_vocab = np.array(vec.get_feature_names_out())
    # In liblinear binary classification, coef_ shape (1, n_features)
    # We extract the slice corresponding to text features by looking at vectorizer dimension
    n_num = len(NUMERIC_FEATS)
    n_txt = len(text_vocab)
    coefs = clf.coef_[0][-n_txt:]  # last n_txt correspond to text block for this setup
    top_pos_idx = np.argsort(coefs)[-15:][::-1]
    top_neg_idx = np.argsort(coefs)[:15]

    imp = pd.DataFrame({
        "feature": list(text_vocab[top_pos_idx]) + list(text_vocab[top_neg_idx]),
        "coef": list(coefs[top_pos_idx]) + list(coefs[top_neg_idx])
    })
    imp["sign"] = ["positive"]*15 + ["negative"]*15
    plt.figure(figsize=(9,6))
    sns.barplot(data=imp, x="coef", y="feature", hue="sign")
    plt.title("Top indicative text n-grams (LogReg)")
    plt.tight_layout()
    txt_imp_path = OUTPUT_DIR/"text_feature_importance.png"
    plt.savefig(txt_imp_path, dpi=160)
    plt.close()
    print(f"Saved text feature importance to: {txt_imp_path}")
except Exception as e:
    print("Could not compute text feature importance:", e)

print("\nDone.")
