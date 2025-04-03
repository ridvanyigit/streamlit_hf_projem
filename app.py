import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# --- Ön İşleme Fonksiyonları (Notebook'ta TF-IDF öncesi yapılıyor olabilir, ama burada explicit yapmak iyi olabilir) ---
# NOT: TF-IDF pipeline'ı zaten küçük harfe çevirme ve tokenizasyon yapıyor olabilir.
# Ancak emin olmak veya daha fazla kontrol için bu adımları ekleyebiliriz.
# Pipeline'ın nasıl eğitildiğine bağlı olarak bu fonksiyonlara ihtiyaç olmayabilir veya
# pipeline'ın beklentisiyle tam uyumlu hale getirilmesi gerekebilir.
# Şimdilik, pipeline'ın temel temizliği yaptığını varsayarak daha basit tutalım.
# Eğer sonuçlar beklenenden farklı olursa, bu fonksiyonları devreye sokup test edebiliriz.

# --- Model Yükleme ---
MODEL_PATH = 'spam_classifier.pkl' # Model dosyasının adı

@st.cache_resource # Modelin tekrar tekrar yüklenmesini önle
def load_model(path):
    """Verilen yoldan pickle modelini yükler."""
    try:
        with open(path, 'rb') as file:
            model = pickle.load(file)
        return model
    except FileNotFoundError:
        st.error(f"Hata: Model dosyası '{path}' bulunamadı. Dosyanın doğru yerde olduğundan emin olun.")
        return None
    except Exception as e:
        st.error(f"Model yüklenirken bir hata oluştu: {e}")
        return None

# Modeli yükle
model = load_model(MODEL_PATH)

# --- Streamlit Arayüzü ---
st.set_page_config(page_title="Spam Classifier", page_icon="📧")

st.title("📧 Spam Mesaj Sınıflandırıcı")
st.write("Girdiğiniz mesajın spam olup olmadığını sınıflandırmak için aşağıdaki metin alanını kullanın.")

message_input = st.text_area("Mesajınızı buraya girin:", height=150, placeholder="Örnek: Click this link to win a prize!")

classify_button = st.button("Mesajı Sınıflandır")

# --- Sınıflandırma Mantığı ---
if classify_button and model is not None:
    if message_input.strip(): # Girdinin boş olup olmadığını kontrol et
        try:
            # Model bir liste veya dizi bekler, tek bir mesajı listeye koy
            prediction = model.predict([message_input])
            result = prediction[0] # Tahmin dizisinden ilk sonucu al ('ham' veya 'spam')

            st.subheader("Sonuç:")
            if result == 'spam':
                st.error(f"🚨 Bu mesaj **SPAM** olarak sınıflandırıldı.")
            else:
                st.success(f"✅ Bu mesaj **HAM** (Spam Değil) olarak sınıflandırıldı.")

        except Exception as e:
            st.error(f"Sınıflandırma sırasında bir hata oluştu: {e}")
            st.error("Lütfen modelin doğru yüklendiğinden ve girdinizin uygun olduğundan emin olun.")

    else:
        st.warning("Lütfen sınıflandırmak için bir mesaj girin.")

elif classify_button and model is None:
    # Model yüklenemediği için butona basılsa bile uyarı ver
    st.error("Model yüklenemediği için sınıflandırma yapılamıyor.")

st.sidebar.info(
    "Bu uygulama, eğitilmiş bir makine öğrenmesi modelini kullanarak "
    "metin mesajlarını 'Spam' veya 'Ham' olarak sınıflandırır."
)