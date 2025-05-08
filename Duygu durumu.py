import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import cv2
import joblib
from kaggle.api.kaggle_api_extended import KaggleApi

# Veri seti indirme
def download_dataset():
    api = KaggleApi()
    api.authenticate()
    dataset_path = "fer2013"
    if not os.path.exists(dataset_path):
        print("Veri seti indiriliyor...")
        api.dataset_download_files('msambare/fer2013', path=dataset_path, unzip=True)
        print("Veri seti başarıyla indirildi!")
    else:
        print("Veri seti zaten mevcut.")

# Veri setini yyükleme ve çalıştırma
def prepare_data():
    csv_file = "C:/Users/emine/.kaggle/fer2013.csv"  # Dosyayı kaydettiğiniz yol
    print("Veriler işleniyor...")
    try:
        data = pd.read_csv(csv_file)
    except FileNotFoundError:
        print("CSV dosyası bulunamadı, lütfen yolu kontrol edin.")
        return None, None, None, None

    print(f"Veri seti başarıyla yüklendi! İlk birkaç örnek:\n{data.head()}")

    # Duygu etiketlerini sadeleştirme bunu kaggle dosyasındaki resimleri inceleyeyek çoğaltabilirsiniz
    emotion_mapping = {
        0: "mutlu", 1: "mutlu", 2: "uzgun", 3: "mutlu", 
        4: "uzgun", 5: "sinirli", 6: "sinirli"
    }
    data['emotion'] = data['emotion'].map(emotion_mapping)

    print(f"Etiketler ve sayılar:\n{data['emotion'].value_counts()}")

    # Piksel verilerini hazırlama
    X = np.array([np.fromstring(pixels, sep=' ') for pixels in data['pixels']])
    X = X / 255.0  # Normalizasyon
    y = data['emotion']
    return train_test_split(X, y, test_size=0.2, random_state=42)

# Modülü eğitme 
def train_model(X_train, X_test, y_train, y_test):
    print("Model eğitiliyor...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Performansı değerlendirme
    y_pred = model.predict(X_test)
    print("Doğruluk Skoru:", accuracy_score(y_test, y_pred))
    print("Sınıflandırma Raporu:\n", classification_report(y_test, y_pred))

    # Modeli kaydetme
    model_path = "duygu_modeli.pkl"
    joblib.dump(model, model_path)
    print(f"Model {model_path} dosyasına kaydedildi.")
    return model_path

# OPENCV İLE DUYGU ANALİZİİ KISMI 
def real_time_emotion_analysis(model_path):
    print("Gerçek zamanlı duygu analizi başlatılıyor...")
    model = joblib.load(model_path)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

        for (x, y, w, h) in faces:
            face = gray[y:y+h, x:x+w]
            face_resized = cv2.resize(face, (48, 48)).flatten().reshape(1, -1) / 255.0  
            emotion = model.predict(face_resized)[0]

            # Çizim ve duygu etiketi
            cv2.putText(frame, emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        cv2.imshow("Duygu Analizi", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Tüm süreci çalıştırma
if __name__ == "__main__":
    # Kaggle API ayarı: .kaggle dizinini kontrol edin
    os.environ['KAGGLE_CONFIG_DIR'] = r"C:/Users/emine/.kaggle"

    # Adımları sırayla çalıştırma
    download_dataset()
    X_train, X_test, y_train, y_test = prepare_data()
    
    if X_train is not None and X_test is not None:
        model_path = train_model(X_train, X_test, y_train, y_test)
        real_time_emotion_analysis(model_path)
