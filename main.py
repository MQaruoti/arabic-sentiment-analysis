import os
import re
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report


# =========================
# 1. Data Loading
# =========================

def read_file_with_encoding(file_path):
    """Read file trying multiple Arabic encodings."""
    encodings = ["utf-8", "windows-1256", "cp1256"]
    for enc in encodings:
        try:
            with open(file_path, "r", encoding=enc) as f:
                return f.read()
        except UnicodeDecodeError:
            continue
    return None


def load_folder(folder_path, label):
    """Load all text files from a folder and assign label."""
    texts = []
    for file in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file)
        content = read_file_with_encoding(file_path)
        if content:
            texts.append(content)

    return pd.DataFrame({"text": texts, "label": label})


def load_dataset(positive_path, negative_path):
    """Combine positive and negative datasets."""
    pos_df = load_folder(positive_path, 1)
    neg_df = load_folder(negative_path, 0)

    df = pd.concat([pos_df, neg_df])
    df = df.sample(frac=1).reset_index(drop=True)  # shuffle
    return df


# =========================
# 2. Text Cleaning
# =========================

def clean_text(text):
    text = re.sub(r'[ًٌٍَُِّْـ]', '', text)  # remove tashkeel
    text = re.sub(r'http\S+', '', text)     # remove URLs
    text = re.sub(r'[^\w\s]', '', text)     # remove punctuation
    text = re.sub(r'\d+', '', text)         # remove numbers
    return text


def preprocess_dataframe(df):
    df["clean_text"] = df["text"].apply(clean_text)
    df["length"] = df["text"].apply(len)
    return df


# =========================
# 3. Train/Test Split
# =========================

def split_data(df, test_size=0.2):
    return train_test_split(
        df["clean_text"],
        df["label"],
        test_size=test_size,
        random_state=42
    )


# =========================
# 4. Model Creation
# =========================

def build_model():
    model = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=5000, ngram_range=(1,2))),
        ("clf", LogisticRegression(class_weight="balanced", max_iter=1000))
    ])
    return model


# =========================
# 5. Training & Evaluation
# =========================

def train_model(model, X_train, y_train):
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))
    return y_pred


# =========================
# 6. Prediction Function
# =========================

def predict_sentiment(model, sentences):
    cleaned = [clean_text(s) for s in sentences]
    predictions = model.predict(cleaned)
    probabilities = model.predict_proba(cleaned)

    for s, p, prob in zip(sentences, predictions, probabilities):
        sentiment = "Positive" if p == 1 else "Negative"
        print(f"Sentence: {s}")
        print(f"Prediction: {sentiment}")
        print(f"Confidence: {prob}")
        print("-" * 40)


# =========================
# 7. Vocabulary Check
# =========================

def check_word_in_vocab(model, word):
    vectorizer = model.named_steps["tfidf"]
    exists = word in vectorizer.vocabulary_
    print(f"Is '{word}' in vocabulary? {exists}")


# =========================
# MAIN EXECUTION
# =========================

positive_path = r"C:\Users\mahmo\Downloads\Dataset\Twitter Data Set (Ar-Twitter)\Positive"
negative_path = r"C:\Users\mahmo\Downloads\Dataset\Twitter Data Set (Ar-Twitter)\Negative"


df = load_dataset(positive_path, negative_path)
df = preprocess_dataframe(df)

X_train, X_test, y_train, y_test = split_data(df)

model = build_model()
model = train_model(model, X_train, y_train)

y_pred = evaluate_model(model, X_test, y_test)

# Test custom sentences
test_sentences = [
    "هذا المنتج ممتاز جداً",
    "خدمة العملاء سيئة للغاية",
    "انتي جميلة"
]

predict_sentiment(model, test_sentences)

check_word_in_vocab(model, "انتي")
