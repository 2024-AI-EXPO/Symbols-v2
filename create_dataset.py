import os
import cv2
import time
import numpy as np
import mediapipe as mp

actions = ["hello", "thanks", "sorry", "hate", "hungry",
           "sick", "tired", "mind", "person", "think",
           "friend", "school", "police", "rice", "bed"]
seq_length = 5
secs_for_action = 30

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    max_num_hands=2,
    min_tracking_confidence=0.5,
    min_detection_confidence=0.5
)

cap = cv2.VideoCapture(0)

file = "./dataset/"
os.makedirs(file, exist_ok=True)


def get_angle(vec, vec_size, i):
    vec = vec / vec_size[:, np.newaxis]

    angle = np.arccos(
        np.einsum(
            'nt,nt->n',
            vec[[0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18], :],
            vec[[1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19], :]
        )
    )
    angle = np.degrees(angle)

    angle_label = np.array([angle], dtype=np.float32)
    angle_label = np.append(angle_label, i)

    return angle_label


while cap.isOpened():
    for idx, action in enumerate(actions):
        ret, frame = cap.read()
        data = []

        frame = cv2.flip(frame, 1)
        cv2.putText(
            frame,
            text=f"{action.upper()}",
            org=(10, 30),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=1,
            color=(255, 255, 255),
            thickness=3
        )
        cv2.imshow("YOUR_FRAME_NAME", frame)
        cv2.waitKey(15000)

        start_time = time.time()

        while time.time() - start_time < secs_for_action:
            ret, frame = cap.read()

            frame = cv2.flip(frame, 1)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = hands.process(frame)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            if result.multi_hand_landmarks is not None:
                ld, rd = [], []
                l_hl, r_hl = [], []
                l_vector, r_vector = [], []
                for res, handed in zip(result.multi_hand_landmarks, result.multi_handedness):
                    joint = np.zeros((21, 4))
                    for j, lm in enumerate(res.landmark):
                        joint[j] = [lm.x, lm.y, lm.x, lm.visibility]

                    v1 = joint[[0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 0, 13, 14, 15, 0, 17, 18, 19], :3]
                    v2 = joint[[i for i in range(1, 21)], :3]
                    v = v2 - v1

                    vector_size = np.linalg.norm(v, axis=1)  # 벡터의 크기
                    if handed.classification[0].label == 'Left':
                        l_vector.append(vector_size)
                        l_hl.append(res)
                        ld.append(np.concatenate([joint.flatten(), get_angle(v, vector_size, idx)]))
                    else:
                        r_vector.append(vector_size)
                        r_hl.append(res)
                        rd.append(np.concatenate([joint.flatten(), get_angle(v, vector_size, idx)]))

                if l_vector:
                    l_vector = np.array(l_vector)
                    freq_left = np.argmax(np.bincount(np.argmax(l_vector, axis=0)))
                    mp_drawing.draw_landmarks(frame, l_hl[freq_left], mp_hands.HAND_CONNECTIONS)
                    data.append(ld[freq_left])
                if r_vector:
                    r_vector = np.array(r_vector)
                    freq_right = np.argmax(np.bincount(np.argmax(r_vector, axis=0)))
                    mp_drawing.draw_landmarks(frame, r_hl[freq_right], mp_hands.HAND_CONNECTIONS)
                    data.append(rd[freq_right])

            cv2.imshow('YOUR_FRAME_NAME', frame)
            if cv2.waitKey(10) == 32:
                break

        data = np.array(data)
        # np.save(os.path.join(file, f'raw_{action}'), data)

        full_seq_data = []
        for seq in range(len(data) - seq_length):
            full_seq_data.append(data[seq:seq + seq_length])
        full_seq_data = np.array(full_seq_data)
        print(action, full_seq_data.shape)
        np.save(os.path.join(file, f'seq_{action}'), full_seq_data)
    break
