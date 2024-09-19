from keras import layers
from keras import Sequential
from keras.api.layers import Dense
from keras.src.saving import load_model
from sklearn.model_selection import train_test_split
from keras.api.callbacks import ModelCheckpoint, ReduceLROnPlateau
import numpy as np
import matplotlib.pyplot as plt

# 추가 데이터
actions = ["hello", "thanks", "sorry", "hate", "hungry",
           "sick", "tired", "mind", "person", "think",
           "friend", "school", "police", "rice", "bed"]
dataset_path = "./dataset/"
data = np.concatenate([np.load(dataset_path + f"seq_{action}.npy") for action in actions], axis=0)

x_data = data[:, :, :-1].astype(np.float32)
y_data = data[:, 0, -1].astype(np.float32)

x_train, x_val, y_train, y_val = train_test_split(x_data, y_data, test_size=0.2, random_state=2024)

# pre_trained_model 불러오기
pre_model = load_model("./models/test_LSTM.keras")

model = Sequential()
for layer in pre_model.layers[:-1]:
    model.add(layer)
model.add(Dense(len(actions), activation="softmax", name="output"))

for layer in model.layers:
    if not isinstance(layer, layers.Dense):
        layer.trainable = False

model.build(input_shape=pre_model.inputs[0].shape)

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model.summary()

# fine_tuning
history = model.fit(
    x_data,
    y_data,
    validation_data=(x_val, y_val),
    epochs=35,
    batch_size=32,
    callbacks=[
        ModelCheckpoint("./models/fine_tuning_LSTM.keras", verbose=2, save_best_only=True, mode="auto"),
        ReduceLROnPlateau(factor=0.5, patience=50, verbose=2, mode="auto")
    ]
)

fig, loss_ax = plt.subplots(figsize=(12, 5))
acc_ax = loss_ax.twinx()

loss_ax.plot(history.history["loss"], "y", label="train_loss")
loss_ax.plot(history.history["val_loss"], "r", label="val_loss")
loss_ax.set_xlabel("epoch")
loss_ax.set_ylabel("loss")
loss_ax.legend(loc="upper right")

acc_ax.plot(history.history["accuracy"], "b", label="train_acc")
acc_ax.plot(history.history["val_accuracy"], "g", label="val_acc")
acc_ax.set_ylabel("accuracy")
acc_ax.legend(loc="center right")
plt.title("LSTM")
plt.ylim(0.0, 1.1)
plt.show()
