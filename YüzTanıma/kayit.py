import cv2
import face_recognition
import pickle
import os

# Kamera başlat
video_capture = cv2.VideoCapture(0)

isim = input("Kişinin adı: ")
print(f"Veri kaydetmek için {isim} yüzünü algılamaya çalışıyoruz...")

yuz_encoding = None

while True:
    ret, frame = video_capture.read()
    rgb_frame = frame[:, :, ::-1]

    # Yüzlerin tam tespit bilgilerini al
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    if face_encodings:
        yuz_encoding = face_encodings[0]

        # Yüzleri kare içine al
        top, right, bottom, left = face_locations[0]
        cv2.rectangle(frame, (left, top), (right, bottom), (255, 0, 0), 2)
        cv2.putText(frame, "Kaydedildi!", (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    cv2.imshow("Kayit", frame)

    # Yüz algılandıysa çık
    if yuz_encoding is not None:
        print("Yüz algılandı, çıkıyoruz...")
        break

    # Manuel çıkış
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Kamera işlemi iptal edildi.")
        break

video_capture.release()
cv2.destroyAllWindows()

# 📁 Klasör yoksa oluştur
klasor = "veriler"
os.makedirs(klasor, exist_ok=True)
print(f"Veriler kaydedilecek klasör: {klasor}")

veri_yolu = os.path.join(klasor, "yuz_verileri.pkl")

# Başlangıçta veri boş
veri = {}

# 📥 Eski verileri yükle (varsa)
if os.path.exists(veri_yolu):
    try:
        with open(veri_yolu, "rb") as f:
            veri = pickle.load(f)
            print("Eski veriler yüklendi.")
    except Exception as e:
        print(f"Eski veriler yüklenirken bir hata oluştu: {e}")
else:
    print("Veri dosyası bulunamadı, yeni dosya oluşturulacak.")

# 🔐 Yeni yüz verisini ekle
print(f"Yeni yüz verisi {isim} ekleniyor...")
veri[isim] = yuz_encoding

# 💾 Yeni veriyi dosyaya kaydet
try:
    print("Veri kaydediliyor...")
    with open(veri_yolu, "wb") as f:
        pickle.dump(veri, f)
    print(f"✅ '{isim}' başarıyla kaydedildi!")
except Exception as e:
    print(f"❌ Kayıt sırasında hata oluştu: {e}")
