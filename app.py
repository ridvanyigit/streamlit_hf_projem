# app.py

import streamlit as st
import pickle
import string
# import nltk # Şu anda direkt kullanılmıyor, pipeline içinde olabilir
# from nltk.corpus import stopwords
# from nltk.stem import PorterStemmer

# --- SAYFA YAPILANDIRMASI (İLK Streamlit Komutu OLMALI) ---
# Bu satır, diğer tüm st.* komutlarından önce gelmelidir.
st.set_page_config(page_title="Spam Classifier", page_icon="📧")

# --- Model Yükleme ---
MODEL_PATH = 'spam_classifier.pkl' # Model dosyasının adı

@st.cache_resource # Modelin tekrar tekrar yüklenmesini önle
def load_model(path):
    """Verilen yoldan pickle modelini yükler."""
    try:
        with open(path, 'rb') as file:
            # Pickle dosyasının sklearn.pipeline.Pipeline içerdiğini varsayıyoruz
            # Eğer sadece model ise ve TF-IDF ayrıysa, burası değişebilir.
            # Ancak notebook'taki kod clf (pipeline) olarak kaydediyor.
            model = pickle.load(file)
        return model
    except FileNotFoundError:
        # Model yüklenemezse hata göster
        st.error(f"Hata: Model dosyası '{path}' bulunamadı. Lütfen dosyanın depoda olduğundan ve adının doğru yazıldığından emin olun.")
        return None
    except ModuleNotFoundError as e:
        st.error(f"Hata: Modeli yüklemek için gerekli kütüphane bulunamadı: {e}. 'requirements.txt' dosyasını kontrol edin (örn: scikit-learn).")
        return None
    except Exception as e:
        st.error(f"Model yüklenirken beklenmedik bir hata oluştu: {e}")
        return None

# Modeli yükle
model = load_model(MODEL_PATH)

# --- Streamlit Arayüzü ---

st.title("📧 Spam Mesaj Sınıflandırıcı")
st.write("Girdiğiniz mesajın spam olup olmadığını sınıflandırmak için aşağıdaki metin alanını kullanın.")

# --- Model Yükleme Kontrolü ---
# Eğer model başarılı bir şekilde yüklenmediyse, uygulamanın geri kalanını çalıştırma.
if model is None:
    st.warning("Model yüklenemediği için uygulama şu anda kullanılamıyor. Lütfen daha sonra tekrar deneyin veya yönetici ile iletişime geçin.")
    st.stop() # Scriptin geri kalanının çalışmasını durdurur

# --- Kullanıcı Girdisi ---
message_input = st.text_area("Mesajınızı buraya girin:", height=150, placeholder="Örnek: Click this link to win a prize!")

# --- Sınıflandırma Butonu ve Mantığı ---
classify_button = st.button("Mesajı Sınıflandır")

if classify_button:
    # Girdinin boş olup olmadığını kontrol et
    if message_input and message_input.strip():
        try:
            # Model bir liste veya dizi bekler, tek bir mesajı listeye koy
            # Pipeline (clf) hem TF-IDF dönüşümünü hem de sınıflandırmayı yapar.
            prediction = model.predict([message_input])
            result = prediction[0] # Tahmin dizisinden ilk sonucu al ('ham' veya 'spam')

            st.subheader("Sonuç:")
            if result == 'spam':
                st.error(f"🚨 Bu mesaj **SPAM** olarak sınıflandırıldı.")
            else:
                st.success(f"✅ Bu mesaj **HAM** (Spam Değil) olarak sınıflandırıldı.")

        except Exception as e:
            st.error(f"Sınıflandırma sırasında bir hata oluştu: {e}")
            st.error("Lütfen girdinizi kontrol edin veya daha sonra tekrar deneyin.")
    else:
        # Kullanıcı butona bastı ama mesaj girmemişse
        st.warning("Lütfen sınıflandırmak için bir mesaj girin.")

# --- Kenar Çubuğu (Sidebar) ---
st.sidebar.header("Uygulama Hakkında")
st.sidebar.info(
    "Bu uygulama, Scikit-learn kullanılarak eğitilmiş bir "
    "makine öğrenmesi modeli (TF-IDF + Random Forest) ile "
    "metin mesajlarını 'Spam' veya 'Ham' olarak sınıflandırır."
)
st.sidebar.markdown("---") # Ayırıcı çizgi
st.sidebar.markdown("Kod [GitHub](https://github.com/ridvanyigit/streamlit_hf_projem)'da bulunmaktadır.") # Kendi repo linkinizi ekleyebilirsiniz