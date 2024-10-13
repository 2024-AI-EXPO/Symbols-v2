import os
import cv2
import numpy as np
import mediapipe as mp
import argparse


def dataset(args):
    # 초기 기본 세팅
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    hands = mp_hands.Hands(
        max_num_hands=3,
        model_complexity=1,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6
    )

    setting: int | str = ""
    file_name = ""
    if args.create == "video":
        setting = args.file
        file_name = args.file.split(".")[0]
        file_name = file_name.split("/")[-1]
    elif args.create == "live":
        setting = 0
        file_name = args.label
    print(file_name)
    cap = cv2.VideoCapture(setting)  # 수정 가능
    seq_len = 20  # 수정 가능 20개의 프레임마다 비교하기 -> fps에 따라 달라질 듯....ㅋㅋㅋ
    data = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            if args.create == "video":
                print("비디오 종료")
                break
            raise Exception("카메라가 제대로 작동하지 않습니다.")
        frame = cv2.flip(frame, 1)
        result = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        if result.multi_hand_landmarks:
            ld, rd = [], []
            l_hl, r_hl = [], []
            l_vector, r_vector = [], []
            for res, handed in zip(result.multi_hand_landmarks, result.multi_handedness):
                joint = np.zeros((21, 3))
                for j, lm in enumerate(res.landmark):
                    joint[j] = [lm.x, lm.y, lm.z]

                v1 = joint[[0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 0, 13, 14, 15, 0, 17, 18, 19]]
                v2 = joint[[i for i in range(1, 21)]]
                v = v2 - v1

                vector_size = np.linalg.norm(v, axis=1)
                dire = v / vector_size[:, np.newaxis]

                angle = np.degrees(
                    np.arccos(
                        np.einsum(
                            "nt,nt->n",
                            dire[[0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18], :],
                            dire[[1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19], :]
                        )
                    )
                )

                angle_label = np.array([angle], dtype=np.float32)
                idx = len(os.listdir(args.directory))
                angle_label = np.append(angle_label, idx)

                if handed.classification[0].label == 'Left':
                    l_vector.append(vector_size)
                    l_hl.append(res)
                    ld.append(np.concatenate([joint.flatten(), angle_label]))
                else:
                    r_vector.append(vector_size)
                    r_hl.append(res)
                    rd.append(np.concatenate([joint.flatten(), angle_label]))

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

        cv2.imshow("result", frame)
        if cv2.waitKey(1) == ord("q"):
            break

    data = np.array(data)
    seq_data = np.array([data[seq:seq+seq_len] for seq in range(len(data) - seq_len)])
    print(file_name, seq_data.shape)
    np.save(os.path.join(args.save, f"{file_name}"), seq_data)


def main():
    parser = argparse.ArgumentParser(description="You can select options for How to make dataset")
    parser.add_argument("-c", "--create", type=str, required=True, help="데이터셋을 영상(video)으로 만들 것인지 실시간(live)으로 만들 것인지 설정하세요 (기본 : 실시간)")
    parser.add_argument("-s", "--save", type=str, required=True, help="데이터셋을 어디에 저장할 지 설정하세요")
    parser.add_argument("-f", "--file", type=str, help="영상으로 데이터셋을 만들 경우 영상 경로를 설정하세요 (파일 이름으로 label) 설정")
    parser.add_argument("-l", "--label", type=str, help="실시간으로 데이터셋을 만들 경우 label을 설정하세요")

    args = parser.parse_args()

    if args.create != "video" and args.create != "live":
        raise Exception("You should select video or live")

    if args.create == "video" and args.file is None:
        raise Exception("Not found directory path for using video")

    if args.create == "live" and args.label is None:
        raise Exception("Not select label")

    dataset(args)


if __name__ == "__main__":
    main()
