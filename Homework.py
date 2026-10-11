import cv2
import mediapipe as mp
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7,
)
mp_draw = mp.solutions.drawing_utils
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not access the webcam.")
    exit()
print("Hand Tracking Started! Press 'q' to quit.")
while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: failed to capture image")
        break
    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            xs = [int(lm.x * w) for lm in hand_landmarks.landmark]
            ys = [int(lm.y * h) for lm in hand_landmarks.landmark]
            pad = 20
            x_min, x_max = max(min(xs) - pad, 0), min(max(xs) + pad, w)
            y_min, y_max = max(min(ys) - pad, 0), min(max(ys) + pad, h)
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
    cv2.imshow("Hand position detected using rectangle", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()