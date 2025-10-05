import os
import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

def train_model_from_csv(data_csv_path="data/train.csv", out_dir="models"):
    os.makedirs(out_dir, exist_ok=True)
    df = pd.read_csv(data_csv_path)

    # Clean column names
    df.columns = [c.lower().strip() for c in df.columns]

    # Detect text + label columns
    text_col = 'text' if 'text' in df.columns else 'content'
    label_col = 'label'

    # Combine title + text
    if 'title' in df.columns:
        df['content'] = df['title'].fillna('') + '. ' + df[text_col].fillna('')
    else:
        df['content'] = df[text_col].astype(str)

    X = df['content'].values
    y = df[label_col].str.upper().values  # normalize labels (REAL/FAKE)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

    vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1,2), max_df=0.9, max_features=5000)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    model = LogisticRegression(max_iter=2000, class_weight='balanced')
    model.fit(X_train_tfidf, y_train)

    preds = model.predict(X_test_tfidf)
    print("✅ Accuracy:", accuracy_score(y_test, preds))
    print(classification_report(y_test, preds, digits=4))

    joblib.dump(model, os.path.join(out_dir, "model.joblib"))
    joblib.dump(vectorizer, os.path.join(out_dir, "vectorizer.joblib"))
    print(f"Model saved in {out_dir}")

if __name__ == "__main__":
    train_model_from_csv("data/train.csv", "models")
