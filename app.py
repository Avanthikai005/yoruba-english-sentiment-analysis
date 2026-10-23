"""Interactive Streamlit demo for the trained sentiment-classification models.

Run from the repository root:
    streamlit run app.py
"""

from pathlib import Path
import pickle

import streamlit as st
import torch

from preprocessing import clean_text, encode_texts, MAX_LEN
from cnn_baseline import TextCNN
from cnn_bilstm_attention import CNNBiLSTMAttention


ROOT = Path(__file__).resolve().parent
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATASETS = {
    "English (SST-2)": "sst2",
    "Yoruba (YOSM)": "yosm",
}
MODELS = {
    "TextCNN": ("cnn_baseline.pt", TextCNN),
    "CNN–BiLSTM–Attention": ("cnn_bilstm_attention.pt", CNNBiLSTMAttention),
}
EXAMPLES = {
    "English (SST-2)": "This movie was beautifully acted and genuinely moving.",
    "Yoruba (YOSM)": "Fiimu yi dara pupọ, mo gbadun rẹ pupọ.",
}


@st.cache_resource(show_spinner=False)
def load_model(dataset: str, model_name: str):
    """Load one locally saved checkpoint and its vocabulary."""
    checkpoint_name, model_class = MODELS[model_name]
    data_dir = ROOT / "data" / "processed" / dataset
    checkpoint_path = ROOT / "results" / dataset / checkpoint_name

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    with (data_dir / "vocab.pkl").open("rb") as file:
        vocab = pickle.load(file)
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    config = checkpoint.get("config", {})
    vocab_size = checkpoint.get("vocab_size", max(vocab.values()) + 1)

    if model_class is TextCNN:
        model = model_class(
            vocab_size,
            emb_dim=config.get("emb_dim", 200),
            kernels=tuple(config.get("kernels", (2, 3, 4, 5))),
            filters=config.get("filters", 100),
        )
    else:
        model = model_class(
            vocab_size,
            emb_dim=config.get("emb_dim", 200),
            kernels=tuple(config.get("kernels", (2, 3, 4, 5))),
            filters=config.get("filters", 100),
            hidden_dim=config.get("hidden_dim", 128),
        )

    model.load_state_dict(checkpoint["state_dict"])
    model.to(DEVICE).eval()
    return model, vocab


def predict(text: str, dataset: str, model_name: str):
    """Clean, encode, and classify one review."""
    model, vocab = load_model(dataset, model_name)
    cleaned = clean_text(text)
    if not cleaned:
        raise ValueError("Please enter a review containing text.")

    encoded = encode_texts([cleaned], vocab, MAX_LEN)
    inputs = torch.from_numpy(encoded).long().to(DEVICE)
    with torch.no_grad():
        probability = torch.sigmoid(model(inputs)).item()
    return cleaned, probability


def main():
    st.set_page_config(page_title="Yoruba–English Sentiment Demo", page_icon="💬", layout="centered")
    st.title("💬 Yoruba–English Sentiment Analysis")
    st.caption("A local demo of TextCNN and CNN–BiLSTM–Attention trained on the project checkpoints.")

    with st.sidebar:
        st.header("Model settings")
        language = st.selectbox("Review language / dataset", list(DATASETS))
        model_name = st.selectbox("Model", list(MODELS))
        st.caption(f"Inference device: {DEVICE.type.upper()}")
        if st.button("Use example review"):
            st.session_state.review_text = EXAMPLES[language]

    st.info(
        "This is sentiment classification, not translation. Enter a review in the selected language "
        "and the selected locally trained model predicts positive or negative sentiment."
    )
    text = st.text_area(
        "Enter a movie review",
        key="review_text",
        height=150,
        placeholder="Type a review in English or Yoruba...",
    )

    if st.button("Predict sentiment", type="primary", use_container_width=True):
        try:
            with st.spinner("Loading model and predicting sentiment..."):
                cleaned, probability = predict(text, DATASETS[language], model_name)
        except (FileNotFoundError, ValueError) as error:
            st.error(str(error))
            return

        label = "Positive" if probability >= 0.5 else "Negative"
        confidence = probability if label == "Positive" else 1 - probability
        st.subheader(f"Prediction: {label}")
        st.progress(int(confidence * 100), text=f"Confidence: {confidence:.1%}")

        left, right = st.columns(2)
        left.metric("Positive probability", f"{probability:.1%}")
        right.metric("Negative probability", f"{1 - probability:.1%}")
        with st.expander("View preprocessing output"):
            st.write(cleaned)
            st.caption(f"Sequence length is padded/truncated to {MAX_LEN} tokens.")

    st.divider()
    st.caption(
        "For academic demonstration only. Confidence is the model probability, not a guarantee of correctness."
    )


if __name__ == "__main__":
    main()
