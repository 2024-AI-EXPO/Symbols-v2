import cv2
import mediapipe as mp
import numpy as np

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    max_num_hands=3,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6,
    model_complexity=1
)


def generate_frames():
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            break

        frame = cv2.flip(frame, 1)
        result = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        if result.multi_hand_landmarks is not None:
            left_hl, right_hl = [], []
            left, right = [], []
            for res, handed in zip(result.multi_hand_landmarks, result.multi_handedness):
                joint = np.zeros((21, 4))
                for j, lm in enumerate(res.landmark):
                    joint[j] = [lm.x, lm.y, lm.z, lm.visibility]

                v1 = joint[[0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 0, 13, 14, 15, 0, 17, 18, 19], :3]
                v2 = joint[[i for i in range(1, 21)], :3]
                v = v2 - v1
                vector_size = np.linalg.norm(v, axis=1)  # 벡터의 크기
                if handed.classification[0].label == 'Left':
                    left.append(vector_size)
                    left_hl.append(res)
                else:
                    right.append(vector_size)
                    right_hl.append(res)

            if left:
                left = np.array(left)
                freq_left = np.argmax(np.bincount(np.argmax(left, axis=0)))
                mp_drawing.draw_landmarks(frame, left_hl[freq_left], mp_hands.HAND_CONNECTIONS)
            if right:
                right = np.array(right)
                freq_right = np.argmax(np.bincount(np.argmax(right, axis=0)))
                mp_drawing.draw_landmarks(frame, right_hl[freq_right], mp_hands.HAND_CONNECTIONS)

        cv2.imshow('', frame)
        if cv2.waitKey(100) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    generate_frames()
