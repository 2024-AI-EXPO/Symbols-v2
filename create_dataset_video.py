import mediapipe as mp
import cv2
import os
import numpy as np

video = cv2.VideoCapture('./KETI_SL_0000010544.mp4')

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    max_num_hands=2,
    min_tracking_confidence=0.5,
    min_detection_confidence=0.5
)

def get_angle(vec, vec_size):
    vec = vec / vec_size[:, np.newaxis]

    angle = np.arccos(
        np.einsum(
            'nt,nt->n',
            vec[[0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18], :],
            vec[[1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19], :]
        )
    )
    return np.degrees(angle)


while video.isOpened():
    ret, frame = video.read()
    
    if not ret:
        print('비디오 오류')
        break

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
                ld.append(np.concatenate([joint.flatten(), get_angle(v, vector_size)]))
            else:
                r_vector.append(vector_size)
                r_hl.append(res)
                rd.append(np.concatenate([joint.flatten(), get_angle(v, vector_size)]))

        if l_vector:
            l_vector = np.array(l_vector)
            freq_left = np.argmax(np.bincount(np.argmax(l_vector, axis=0)))
            mp_drawing.draw_landmarks(frame, l_hl[freq_left], mp_hands.HAND_CONNECTIONS)
        if r_vector:
            r_vector = np.array(r_vector)
            freq_right = np.argmax(np.bincount(np.argmax(r_vector, axis=0)))
            mp_drawing.draw_landmarks(frame, r_hl[freq_right], mp_hands.HAND_CONNECTIONS)

    cv2.imshow('YOUR_FRAME_NAME', frame)

    if cv2.waitKey(1) == ord('q'):
        print('동영상 조기종료')
        break

video.release()
cv2.destroyAllWindows()