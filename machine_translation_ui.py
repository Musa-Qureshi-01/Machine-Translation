import streamlit as st
import numpy as np
import pickle
import time
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ---------------------------------------------------
# PAGE CONFIGURATION
# ---------------------------------------------------
st.set_page_config(
    page_title="English → French Translator",
    page_icon="🌍",
    layout="centered"
)

# ---------------------------------------------------
# CUSTOM CSS (Animations + Glassmorphism UI)
# ---------------------------------------------------
st.markdown("""
<style>

body {
    background: linear-gradient(135deg, #dfe9f3, #ffffff);
    font-family: 'Segoe UI', sans-serif;
}

/* Glass UI Card */
.translator-box {
    padding: 30px;
    background: rgba(255, 255, 255, 0.75);
    border-radius: 20px;
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    box-shadow: 0px 8px 25px rgba(0,0,0,0.1);
    border: 1px solid rgba(255,255,255,0.3);
    animation: fadeIn 1s ease-in-out;
}

/* Fade-In Animation */
@keyframes fadeIn {
    from { opacity: 0; transform: translateY(20px); }
    to   { opacity: 1; transform: translateY(0px); }
}

/* Output Box Animation */
.output-box {
    padding: 20px;
    background: #eef7ff;
    border-left: 6px solid #2196F3;
    border-radius: 12px;
    font-size: 18px;
    animation: slideIn 0.6s ease-out;
    margin-top: 10px;
}

/* Slide Animation */
@keyframes slideIn {
    from { opacity: 0; transform: translateX(40px); }
    to   { opacity: 1; transform: translateX(0px); }
}

/* Centered Subtitle */
.subtitle {
    text-align: center;
    font-size: 18px;
    margin-top: -15px;
    color: #555;
}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------
# HELPER — FILE CHECKER
# ---------------------------------------------------
def check_file(path, name):
    """Check if file exists and show user-friendly error"""
    if not os.path.exists(path):
        st.error(f"❌ `{name}` not found.\n\nPlease place `{name}` in the same folder as this app.")
        st.stop()

# ---------------------------------------------------
# LOAD MODEL & TOKENIZERS
# ---------------------------------------------------
@st.cache_resource
def load_translation_model():
    check_file("translation_model.h5", "translation_model.h5")
    return load_model("translation_model.h5")

@st.cache_resource
def load_tokenizer(path, name):
    check_file(path, name)
    with open(path, "rb") as f:
        return pickle.load(f)

# Check files BEFORE loading
check_file("translation_model.h5", "translation_model.h5")
check_file("eng_tokenizer.pkl", "eng_tokenizer.pkl")
check_file("fr_tokenizer.pkl", "fr_tokenizer.pkl")

# Load assets
model = load_translation_model()
eng_tokenizer = load_tokenizer("eng_tokenizer.pkl", "eng_tokenizer.pkl")
fr_tokenizer = load_tokenizer("fr_tokenizer.pkl", "fr_tokenizer.pkl")

# Max sequence lengths
max_len_eng = 20
max_len_fr = 20

# Create index → word mapping
index_to_word_fr = {v: k for k, v in fr_tokenizer.word_index.items()}

# ---------------------------------------------------
# TRANSLATION LOGIC
# ---------------------------------------------------
def translate_sentence(sentence):
    seq = eng_tokenizer.texts_to_sequences([sentence])
    seq = pad_sequences(seq, maxlen=max_len_eng, padding='post')

    preds = model.predict(seq)
    pred_ids = np.argmax(preds[0], axis=1)
    words = [index_to_word_fr.get(i, "") for i in pred_ids]
    return " ".join(words).strip()

# ---------------------------------------------------
# TYPING ANIMATION
# ---------------------------------------------------
def typing_animation(text):
    output = ""
    placeholder = st.empty()

    for char in text:
        output += char
        placeholder.markdown(
            f"<div class='output-box'><strong>{output}</strong></div>",
            unsafe_allow_html=True
        )
        time.sleep(0.02)  # typing speed

# ---------------------------------------------------
# UI LAYOUT
# ---------------------------------------------------
st.markdown("<h1 style='text-align:center;'>🌍 English → French Machine Translation</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Deep Learning · Seq2Seq · Bidirectional LSTMs</p>", unsafe_allow_html=True)

st.write("<br>", unsafe_allow_html=True)

# Main UI Card
st.markdown("<div class='translator-box'>", unsafe_allow_html=True)

user_input = st.text_area(
    "✍️ Enter English Text:",
    height=140,
    placeholder="Type your English sentence here...",
)

btn = st.button("🚀 Translate")

# ---------------------------------------------------
# TRANSLATION TRIGGER
# ---------------------------------------------------
if btn:
    if user_input.strip() == "":
        st.warning("⚠️ Please enter some text to translate.")
    else:
        with st.spinner("Translating with AI model..."):
            result = translate_sentence(user_input)

        st.success("✨ Translation Ready:")
        typing_animation(result)

# End of UI Box
st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)
st.caption("Built with ❤️ using Streamlit · Deep Learning · WMT Dataset")

