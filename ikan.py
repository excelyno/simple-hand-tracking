import cv2
import mediapipe as mp

# Inisialisasi MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Konfigurasi Hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

# Buka kamera
cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("Gagal membaca frame dari kamera.")
        break

    # Konversi ke RGB karena MediaPipe memproses gambar dalam format RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Proses gambar untuk mendeteksi tangan
    results = hands.process(image_rgb)

    # Jika tangan terdeteksi
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Gambar landmark dan koneksi di gambar
            mp_drawing.draw_landmarks(
                image, hand_landmarks, mp_hands.HAND_CONNECTIONS
            )

    # Tampilkan hasil
    cv2.imshow('Hand Detection', image)

    # Berhenti dengan menekan 'q'
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

# Lepaskan sumber daya
cap.release()
cv2.destroyAllWindows()
hands.close()
