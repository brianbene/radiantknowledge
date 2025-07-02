import streamlit as st
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import json
import os
import wikipedia
import torch.nn.functional as F

try:
    APP_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    APP_DIR = os.getcwd()

ARTIFACTS_PATH = os.path.join(APP_DIR, "NuclearAppArtifacts")
BERT_MODEL_PATH = os.path.join(ARTIFACTS_PATH, "BERT_model_final")
LABELS_PATH = os.path.join(ARTIFACTS_PATH, "class_labels.json")
IMAGE_PATH = os.path.join(APP_DIR, "Nuclear_Plant.gif")


@st.cache_resource
def load_model_and_artifacts():
    if not os.path.exists(BERT_MODEL_PATH):
        st.error(
            f"BERT model directory not found at '{BERT_MODEL_PATH}'. Please ensure the 'NuclearAppArtifacts/BERT_model_final' folder structure exists relative to this script.")
        return None, None, None

    if not os.path.exists(LABELS_PATH):
        st.error(
            f"Labels file not found at '{LABELS_PATH}'. Please ensure 'class_labels.json' is inside the 'NuclearAppArtifacts' folder.")
        return None, None, None

    try:
        tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_PATH)
        model = BertForSequenceClassification.from_pretrained(BERT_MODEL_PATH)
    except Exception as e:
        st.error(f"Error loading BERT model: {e}")
        return None, None, None

    with open(LABELS_PATH, 'r') as f:
        class_labels = json.load(f)

    model.eval()
    return model, tokenizer, class_labels


@st.cache_resource
def setup_wikipedia():
    wikipedia.set_user_agent('NuclearDocumentClassifier/1.0 (youremail@example.com)')


def classify_text(text, model, tokenizer, class_labels):
    encoding = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=256,
        return_token_type_ids=False,
        padding='max_length',
        return_attention_mask=True,
        return_tensors='pt',
        truncation=True
    )

    with torch.no_grad():
        outputs = model(
            input_ids=encoding['input_ids'],
            attention_mask=encoding['attention_mask']
        )

    probabilities = F.softmax(outputs.logits, dim=1).flatten()

    confidence, predicted_class_idx = torch.max(probabilities, dim=0)
    predicted_class_label = class_labels[predicted_class_idx]

    return predicted_class_label, confidence.item()


st.set_page_config(page_title="Nuclear Document Classifier", layout="wide")

st.title("Radiant Knowledge Research Tool 🔎")
st.markdown("""
This tool uses a fine-tuned BERT model to classify your text into a relevant nuclear engineering category. 
Based on the prediction, it will then retrieve the most relevant articles from Wikipedia to kickstart your research.
""")

model, tokenizer, class_labels = load_model_and_artifacts()
setup_wikipedia()

if model and tokenizer and class_labels:
    st.subheader("Enter Your Text Query")
    user_query = st.text_area(
        "Paste a paragraph, a question, or a topic here:",
        height=150,
        placeholder="e.g., 'What are the safety protocols for a control rod failure in a pressurized water reactor?'"
    )

    classify_button = st.button("Classify and Find Articles", type="primary")

    # The ratios have been adjusted to make the center column (and thus the image) larger.
    col1, col2, col3 = st.columns([1, 1.5, 1])
    with col2:
        if os.path.exists(IMAGE_PATH):
            st.image(IMAGE_PATH)
        else:
            st.warning(
                f"Image not found")

    if classify_button:
        if user_query:
            with st.spinner("Analyzing text and searching Wikipedia..."):
                predicted_category, confidence = classify_text(user_query, model, tokenizer, class_labels)

                st.subheader("Classification Result")
                res_col1, res_col2 = st.columns(2)
                with res_col1:
                    st.success(f"**Predicted Category:** `{predicted_category}`")
                with res_col2:
                    st.info(f"**Confidence:** {confidence:.2%}")

                st.subheader("Relevant Wikipedia Articles")

                search_term = predicted_category.replace('_', ' ')

                try:
                    search_results = wikipedia.search(search_term, results=3)

                    if search_results:
                        for title in search_results:
                            try:
                                page = wikipedia.page(title, auto_suggest=False, redirect=True)
                                with st.expander(f"**{page.title}**"):
                                    st.markdown(f"**Source:** [{page.url}]({page.url})")
                                    st.write(page.summary)
                            except wikipedia.exceptions.PageError:
                                st.warning(
                                    f"Could not find a specific page for '{title}'. It may have been a search suggestion that doesn't correspond to a real page.")
                            except wikipedia.exceptions.DisambiguationError as e:
                                st.warning(
                                    f"'{title}' is a disambiguation page. The top option is often '{e.options[0]}'.")

                    else:
                        st.warning(f"Could not find any direct Wikipedia articles for the category '{search_term}'.")

                except Exception as e:
                    st.error(f"An error occurred while searching Wikipedia: {e}")
                    st.info(
                        "This might be due to a network issue or a problem with the Wikipedia API. Please try again later.")

        else:
            st.warning("Please enter some text to classify.")
else:
    st.error(
        "Application could not be started. Please check the artifact paths and ensure the model files are correct.")
