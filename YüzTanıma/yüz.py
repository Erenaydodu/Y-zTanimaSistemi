import cv2
import face_recognition

# Kamerayı başlat (0 = varsayılan kamera)
video_capture = cv2.VideoCapture(0)

while True:
    # Kameradan bir kare al
    ret, frame = video_capture.read()

    # Renkleri BGR'den RGB'ye çevir
    rgb_frame = frame[:, :, ::-1]

    # Yüzleri algıla
    face_locations = face_recognition.face_locations(rgb_frame)

    # Her yüz için dikdörtgen çiz
    for top, right, bottom, left in face_locations:
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

    # Görüntüyü göster
    cv2.imshow('Yüz Algılama', frame)

    # 'q' tuşuna basınca çık
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Temizlik
video_capture.release()
cv2.destroyAllWindows()
