# app.py

import streamlit as st
import pickle
import string
# import nltk # Currently not used directly, might be within the pipeline
# from nltk.corpus import stopwords
# from nltk.stem import PorterStemmer
import numpy as np # Needed for type checking in one of the tests if added later
import os # Used in one of the tests if added later

# --- PAGE CONFIGURATION (MUST be the first Streamlit command) ---
st.set_page_config(page_title="Spam Classifier", page_icon="📧")

# --- Model Loading ---
MODEL_PATH = 'spam_classifier.pkl' # Name of the model file

@st.cache_resource # Prevent reloading the model repeatedly
def load_model(path):
    """Loads the pickle model from the given path."""
    try:
        with open(path, 'rb') as file:
            model = pickle.load(file)
        return model
    except FileNotFoundError:
        st.error(f"Error: Model file '{path}' not found. Please ensure the file exists in the repository and the name is correct.")
        return None
    except ModuleNotFoundError as e:
        st.error(f"Error: A required library to load the model was not found: {e}. Check 'requirements.txt' (e.g., scikit-learn).")
        return None
    except Exception as e:
        st.error(f"An unexpected error occurred while loading the model: {e}")
        return None

# Load the model
model = load_model(MODEL_PATH)

# --- Streamlit Interface ---

st.title("📧 Spam Message Classifier")
st.write("Enter your message below to classify it as Spam or Ham.")

# --- Model Loading Check ---
if model is None:
    st.warning("The application is currently unavailable because the model could not be loaded. Please try again later or contact the administrator.")
    st.stop() # Stops execution of the script

# --- User Input ---
message_input = st.text_area("Enter your message here:", height=150, placeholder="Example: Click this link to win a prize!")

# --- Classification Button and Logic ---
classify_button = st.button("Classify Message")

if classify_button:
    if message_input and message_input.strip():
        try:
            prediction = model.predict([message_input])
            result = prediction[0]

            st.subheader("Result:")
            if result == 'spam':
                st.error(f"🚨 This message was classified as **SPAM**.")
            else:
                st.success(f"✅ This message was classified as **HAM** (Not Spam).")

            # --- BALOONS ---
            # Sınıflandırma sonucu gösterildikten hemen sonra çalışır.
            st.balloons()

        except Exception as e:
            st.error(f"An error occurred during classification: {e}")
            st.error("Please check your input or try again later.")
    else:
        st.warning("Please enter a message to classify.")

# --- Sidebar ---
st.sidebar.header("About the App")
st.sidebar.info(
    "This application classifies text messages as 'Spam' or 'Ham' "
    "using a machine learning model (TF-IDF + Random Forest) "
    "trained with Scikit-learn."
)
st.sidebar.markdown("---") # Separator line
st.sidebar.markdown("The code is available on [GitHub](https://github.com/ridvanyigit/streamlit_hf_projem).")
