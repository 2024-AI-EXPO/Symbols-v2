import pandas as pd
import os

label_data = pd.read_excel("/home/modeep/GitHub/Symbols-v2/KETI-2017-SL-Annotation-v2_1.xlsx")
label_data = label_data.loc[:, ~label_data.columns.str.contains('^Unnamed')]
label_data = label_data[label_data["번호"] > 3000]
label_word = label_data[label_data["타입(단어/문장)"] == "단어"]

path = "/media/modeep/711a1c4d-6963-4fb1-a8af-eb3bf2a75f83/수어 데이터/수어 데이터셋/"
data_list = os.listdir(path)
os.makedirs("/home/modeep/GitHub/Symbols-v2/train", exist_ok=True)

for data in data_list:
    video_list = os.listdir(path + data)
    for video in video_list:
        number = int(video.split('_')[2].split('.')[0])
        label = label_word[label_word["번호"] == number]["한국어"].tolist()
        if label:
            with open(f"/home/modeep/GitHub/Symbols-v2/train/{label[0]}.txt", "at") as f:
                f.write(f"{video}\n")
