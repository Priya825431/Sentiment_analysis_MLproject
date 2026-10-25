import re
import os
import html
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, ConfusionMatrixDisplay
)

st.set_page_config(page_title="Sentiment Analysis | Ensemble Learning", page_icon="😊", layout="wide")
st.title("😊 Sentiment Analysis using Ensemble Learning")
st.write("Analyze tweet sentiment using TF-IDF, Random Forest, and AdaBoost.")
st.caption("Dataset: Sentiment140 | Model scores below are calculated from the selected sample and test split.")

LOCAL_DATASET = "training.1600000.processed.noemoticon.csv"
SENTIMENT140_COLUMNS = ["sentiment", "id", "date", "query", "user", "text"]


def clean_text(value):
    text = str(value).lower()
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"@[\w]+", " ", text)
    text = re.sub(r"#", "", text)
    text = re.sub(r"[^a-z\s']", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def read_sentiment140_sample(path, sample_size, chunk_size=100_000):
    """Read the large CSV in chunks and keep a random sample to reduce memory use."""
    chunks = []
    rows_seen = 0
    keep_fraction = min(1.0, sample_size / 1_600_000)
    for chunk in pd.read_csv(
        path, header=None, names=SENTIMENT140_COLUMNS,
        encoding="latin-1", on_bad_lines="skip", chunksize=chunk_size
    ):
        rows_seen += len(chunk)
        # Random sample from each chunk; cap later to requested size.
        part = chunk.sample(frac=keep_fraction, random_state=42) if keep_fraction < 1 else chunk
        chunks.append(part)
        if sum(len(x) for x in chunks) >= sample_size * 1.5:
            break
    if not chunks:
        return pd.DataFrame(columns=SENTIMENT140_COLUMNS), rows_seen
    sampled = pd.concat(chunks, ignore_index=True)
    if len(sampled) > sample_size:
        sampled = sampled.sample(n=sample_size, random_state=42)
    return sampled, rows_seen


with st.sidebar:
    st.header("Settings")
    sample_size = st.select_slider(
        "Number of tweets to use",
        options=[2000, 5000, 10000, 20000, 50000],
        value=5000,
        help="Start with 2,000 or 5,000 tweets. Larger samples take longer to train."
    )
    test_size = st.slider("Test data (%)", min_value=20, max_value=40, value=25, step=5)
    st.info("Tip: start with 5,000 tweets for your first run.")

st.subheader("1. Load the dataset")
source = st.radio(
    "Choose where to load Sentiment140 from:",
    ["Use CSV in project folder", "Upload CSV file"],
    horizontal=True
)

uploaded_file = None
if source == "Use CSV in project folder":
    if os.path.exists(LOCAL_DATASET):
        st.success(f"Found local dataset: `{os.path.abspath(LOCAL_DATASET)}`")
        data_path = LOCAL_DATASET
    else:
        st.warning(f"Could not find `{LOCAL_DATASET}` in the same folder as app.py. Put the CSV next to app.py or choose Upload CSV file.")
        data_path = None
else:
    uploaded_file = st.file_uploader("Upload Sentiment140 CSV", type=["csv", "txt"])
    data_path = uploaded_file
    if uploaded_file is not None:
        st.caption("The original 228 MB file may exceed Streamlit's default upload limit. Using the local-file option avoids that upload limit.")

if data_path is None:
    st.stop()

if st.button("Load dataset sample", type="primary"):
    with st.spinner("Reading the dataset in chunks and preparing a sample..."):
        if isinstance(data_path, str):
            raw_sample, rows_read = read_sentiment140_sample(data_path, sample_size)
            st.session_state["raw_sample"] = raw_sample
            st.session_state["rows_read"] = rows_read
        else:
            # Sentiment140's original CSV has no header. Read only enough rows for a quick run.
            raw_sample = pd.read_csv(
                data_path, header=None, names=SENTIMENT140_COLUMNS,
                encoding="latin-1", on_bad_lines="skip"
            )
            if len(raw_sample) > sample_size:
                raw_sample = raw_sample.sample(n=sample_size, random_state=42)
            st.session_state["raw_sample"] = raw_sample
            st.session_state["rows_read"] = len(raw_sample)
        st.session_state.pop("trained", None)

if "raw_sample" not in st.session_state:
    st.info("Click **Load dataset sample** to continue.")
    st.stop()

raw = st.session_state["raw_sample"].copy()

# Convert original Sentiment140 labels: 0 = negative, 4 = positive.
raw["sentiment"] = raw["sentiment"].astype(str).str.strip()
raw = raw[raw["sentiment"].isin(["0", "2", "4"])].copy()
raw["label"] = raw["sentiment"].map({"0": "Negative", "2": "Neutral", "4": "Positive"})
raw["text"] = raw["text"].fillna("").astype(str)
raw = raw[raw["text"].str.strip().ne("")].drop_duplicates(subset=["text", "label"])

if raw.empty:
    st.error("No usable rows found. Check that you selected the original Sentiment140 CSV without headers.")
    st.stop()

st.subheader("2. Dataset preview and information")
col1, col2, col3 = st.columns(3)
col1.metric("Rows in sample", f"{len(raw):,}")
col2.metric("Columns", str(raw.shape[1]))
col3.metric("Unique tweets", f"{raw['text'].nunique():,}")
st.dataframe(raw[["sentiment", "label", "text"]].head(15), use_container_width=True)
with st.expander("View dataset summary"):
    st.write("Column names:", list(raw.columns))
    st.write("Missing values:")
    st.dataframe(raw.isna().sum().rename("Missing values").to_frame())
    st.write("Sentiment counts:")
    st.dataframe(raw["label"].value_counts().rename("Count").to_frame())

st.subheader("3. Exploratory data analysis")
counts = raw["label"].value_counts().reindex(["Negative", "Neutral", "Positive"]).dropna()
fig, ax = plt.subplots()
counts.plot(kind="bar", ax=ax)
ax.set_title("Sentiment Distribution in Selected Sample")
ax.set_xlabel("Sentiment")
ax.set_ylabel("Number of tweets")
ax.tick_params(axis="x", rotation=0)
st.pyplot(fig)
plt.close(fig)

st.subheader("4. Text preprocessing")
raw["clean_text"] = raw["text"].apply(clean_text)
st.dataframe(raw[["text", "clean_text", "label"]].head(10), use_container_width=True)

# Train only on positive and negative rows, since these are the standard Sentiment140 labels.
model_data = raw[raw["label"].isin(["Negative", "Positive"])].copy()
model_data = model_data[model_data["clean_text"].str.len() > 0]
if model_data["label"].nunique() < 2 or model_data["label"].value_counts().min() < 2:
    st.error("The selected sample needs at least two examples of both Negative and Positive. Load a larger sample.")
    st.stop()

st.subheader("5. Train and evaluate the models")
st.write("TF-IDF converts tweet text into numerical features. The data is split into training and testing sets.")
if st.button("Train Random Forest and AdaBoost", type="primary"):
    progress = st.progress(0, text="Preparing TF-IDF features...")
    X_train_text, X_test_text, y_train, y_test = train_test_split(
        model_data["clean_text"], model_data["label"],
        test_size=test_size / 100, random_state=42, stratify=model_data["label"]
    )
    vectorizer = TfidfVectorizer(max_features=3000, stop_words="english", ngram_range=(1, 2), sublinear_tf=True)
    X_train = vectorizer.fit_transform(X_train_text)
    X_test = vectorizer.transform(X_test_text)
    progress.progress(25, text="Training Random Forest...")
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1, class_weight="balanced")
    rf.fit(X_train, y_train)
    progress.progress(60, text="Training AdaBoost...")
    ada = AdaBoostClassifier(n_estimators=50, random_state=42)
    ada.fit(X_train, y_train)
    progress.progress(85, text="Calculating evaluation metrics...")

    results = {}
    predictions = {}
    for name, model in [("Random Forest", rf), ("AdaBoost", ada)]:
        pred = model.predict(X_test)
        predictions[name] = pred
        results[name] = {
            "Accuracy": accuracy_score(y_test, pred),
            "Precision": precision_score(y_test, pred, average="weighted", zero_division=0),
            "Recall": recall_score(y_test, pred, average="weighted", zero_division=0),
            "F1-score": f1_score(y_test, pred, average="weighted", zero_division=0),
        }
    st.session_state["trained"] = {
        "vectorizer": vectorizer, "rf": rf, "ada": ada,
        "y_test": y_test, "predictions": predictions,
        "results": results, "X_test_text": X_test_text
    }
    progress.progress(100, text="Training complete!")

if "trained" in st.session_state:
    trained = st.session_state["trained"]
    results_df = pd.DataFrame(trained["results"]).T
    st.success("Both models trained successfully.")
    st.markdown("**Evaluation metrics**")
    st.dataframe(results_df.style.format("{:.3f}"), use_container_width=True)

    metric_choice = st.selectbox("Metric to compare", ["Accuracy", "Precision", "Recall", "F1-score"])
    fig, ax = plt.subplots()
    results_df[metric_choice].plot(kind="bar", ax=ax)
    ax.set_title(f"Random Forest vs AdaBoost — {metric_choice}")
    ax.set_ylabel(metric_choice)
    ax.set_ylim(0, 1)
    ax.tick_params(axis="x", rotation=0)
    st.pyplot(fig)
    plt.close(fig)

    st.subheader("6. Confusion matrices")
    cm_cols = st.columns(2)
    for col, name in zip(cm_cols, ["Random Forest", "AdaBoost"]):
        with col:
            fig, ax = plt.subplots()
            ConfusionMatrixDisplay.from_predictions(
                trained["y_test"], trained["predictions"][name],
                labels=["Negative", "Positive"], display_labels=["Negative", "Positive"],
                cmap="Blues", ax=ax, colorbar=False
            )
            ax.set_title(name)
            st.pyplot(fig)
            plt.close(fig)
            with st.expander(f"{name} classification report"):
                st.text(classification_report(trained["y_test"], trained["predictions"][name], zero_division=0))

    st.subheader("7. Predict the sentiment of a new tweet")
    tweet = st.text_area("Enter a tweet", "I love this product, it is amazing!")
    selected_model = st.radio("Choose model", ["Random Forest", "AdaBoost"], horizontal=True)
    if st.button("Predict sentiment"):
        cleaned = clean_text(tweet)
        if not cleaned:
            st.warning("Please enter a tweet containing some text.")
        else:
            features = trained["vectorizer"].transform([cleaned])
            model = trained["rf"] if selected_model == "Random Forest" else trained["ada"]
            prediction = model.predict(features)[0]
            if hasattr(model, "predict_proba"):
                confidence = float(max(model.predict_proba(features)[0]))
                st.metric("Prediction", prediction, f"Model confidence: {confidence:.1%}")
            else:
                st.metric("Prediction", prediction)

    st.subheader("8. Download test predictions")
    export = pd.DataFrame({
        "tweet": trained["X_test_text"].values,
        "actual_sentiment": trained["y_test"].values,
        "random_forest_prediction": trained["predictions"]["Random Forest"],
        "adaboost_prediction": trained["predictions"]["AdaBoost"],
    })
    st.download_button("Download predictions as CSV", export.to_csv(index=False).encode("utf-8"), "sentiment_predictions.csv", "text/csv")
else:
    st.info("Click **Train Random Forest and AdaBoost** after reviewing the dataset and preprocessing samples.")

st.caption("For reproducibility: random_state=42. Results depend on sample size, data cleaning, and test split.")
