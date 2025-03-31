import cv2
import mediapipe as mp
import csv
import os

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Define gestures and CSV file
gestures = ["Hello", "Bye", "I_Love_You", "Thank_You", "Yes", "No"]
csv_filename = "sign_language_data.csv"

# Create CSV file and write header
if not os.path.exists(csv_filename):
    with open(csv_filename, "w", newline="") as file:
        writer = csv.writer(file)
        header = ["Label"] + [f"x{i}" for i in range(21)] + [f"y{i}" for i in range(21)]  # 21 landmarks (x, y)
        writer.writerow(header)

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    cv2.imshow("Collect Sign Language Data", frame)

    key = cv2.waitKey(1) & 0xFF
    if key in [ord('1'), ord('2'), ord('3'), ord('4'), ord('5'), ord('6')]:
        label = gestures[int(chr(key)) - 1]
        data = [label]

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                for lm in hand_landmarks.landmark:
                    data.append(lm.x)
                    data.append(lm.y)

        # Save to CSV
        with open(csv_filename, "a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(data)
            print(f"Saved data for {label}")

    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
