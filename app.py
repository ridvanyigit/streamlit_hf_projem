# app.py

import streamlit as st
import pickle
import string
# import nltk
# from nltk.corpus import stopwords
# from nltk.stem import PorterStemmer
import numpy as np # test_model_classes içinde kullanıldıysa import edilmeli

# --- ZIMANÊ RÛPELA Sazkirin (Divê Fermana Streamlit a YEKEM be) ---
st.set_page_config(page_title="Dabeşkerê Spamê", page_icon="📧") # Rûpel Sernav: Spam Classifier -> Dabeşkerê Spamê

# --- Barkirina Modelê ---
MODEL_PATH = 'spam_classifier.pkl'

@st.cache_resource
def load_model(path):
    """Modela pickle ji rêça diyarkirî bar dike."""
    try:
        with open(path, 'rb') as file:
            model = pickle.load(file)
        return model
    except FileNotFoundError:
        st.error(f"Çewtî: Pelê modelê '{path}' nehat dîtin. Ji kerema xwe piştrast bikin ku pel di depoyê de ye û nav rast hatiye nivîsandin.")
        return None
    except ModuleNotFoundError as e:
        st.error(f"Çewtî: Pirtûkxaneya pêwîst ji bo barkirina modelê nehat dîtin: {e}. Pelê 'requirements.txt' kontrol bikin (mînak: scikit-learn).")
        return None
    except Exception as e:
        st.error(f"Di dema barkirina modelê de çewtiyek nediyar çêbû: {e}")
        return None

# Modelê bar bike
model = load_model(MODEL_PATH)

# --- Navrûya Bikarhêner a Streamlit ---

st.title("📧 Dabeşkerê Peyamên Spam") # Sernav: Spam Mesaj Sınıflandırıcı -> Dabeşkerê Peyamên Spam
st.write("Ji bo dabeşkirina ka peyama we spam e an na, qada nivîsê ya jêrîn bikar bînin.") # Açıklama metni

# --- Kontrola Barkirina Modelê ---
if model is None:
    st.warning("Ji ber ku model nehat barkirin, sepan niha nayê bikaranîn. Ji kerema xwe paşê dîsa biceribînin an bi rêveberê re têkilî daynin.")
    st.stop()

# --- Têketina Bikarhêner ---
message_input = st.text_area(
    "Peyama xwe li vir binivîsin:", # Metin alanı etiketi
    height=150,
    placeholder="Mînak: Click this link to win a prize!" # Placeholder metni (İngilizce bırakmak daha iyi olabilir, model İngilizce bekliyor)
)

# --- Bişkoja Dabeşkirinê û Mantiq ---
classify_button = st.button("Peyamê Dabeş Bike") # Buton metni: Mesajı Sınıflandır -> Peyamê Dabeş Bike

if classify_button:
    if message_input and message_input.strip():
        try:
            # Model navnîşanek an rêzek hêvî dike, peyamek yekane têxe navnîşanê
            prediction = model.predict([message_input])
            result = prediction[0]

            st.subheader("Encam:") # Alt başlık: Sonuç -> Encam
            if result == 'spam':
                st.error(f"🚨 Ev peyam wekî **SPAM** hate dabeş kirin.") # Spam sonucu
            else:
                st.success(f"✅ Ev peyam wekî **HAM** (Ne Spam) hate dabeş kirin.") # Ham sonucu

        except Exception as e:
            st.error(f"Di dema dabeşkirinê de çewtiyek çêbû: {e}") # Sınıflandırma hatası
    else:
        st.warning("Ji kerema xwe ji bo dabeşkirinê peyamek binivîsin.") # Boş mesaj uyarısı


# --- Sidebar (Milê Çepê) ---
st.sidebar.header("Derbarê Sepanê") # Sidebar başlığı: Uygulama Hakkında -> Derbarê Sepanê
st.sidebar.info(
    "Ev sepan, bi karanîna modelek fêrbûna makîneyê (TF-IDF + Random Forest) ya ku bi Scikit-learn hatiye perwerde kirin, "
    "peyamên nivîsê wekî 'Spam' an 'Ham' dabeş dike."
) # Sidebar bilgi metni
st.sidebar.markdown("---")
st.sidebar.markdown("Kod li ser [GitHub](https://github.com/ridvanyigit/streamlit_hf_projem)ê heye.") # Sidebar link metni