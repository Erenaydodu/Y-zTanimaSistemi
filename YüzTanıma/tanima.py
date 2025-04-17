import cv2
import face_recognition
import pickle

# Kayıtlı yüz verilerini yükle
with open("veriler/yuz_verileri.pkl", "rb") as f:
    veri = pickle.load(f)

isimler = list(veri.keys())
yuzler = list(veri.values())

# Kamera başlat
video_capture = cv2.VideoCapture(0)

print("Yüz tanıma başlatıldı. Çıkmak için 'q' tuşuna bas.")

while True:
    ret, frame = video_capture.read()
    rgb_frame = frame[:, :, ::-1]

    # Yüz konumlarını ve encoding'lerini bul
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        # Mevcut yüz ile tüm kayıtlı yüzleri karşılaştır
        matches = face_recognition.compare_faces(yuzler, face_encoding)
        name = "Bilinmeyen"

        # Eşleşme varsa ismi al
        if True in matches:
            first_match_index = matches.index(True)
            name = isimler[first_match_index]

        # Dikdörtgen ve isim çiz
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    cv2.imshow("Yüz Tanıma", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
