from keras import Sequential
from keras.api.layers import LSTM, Input, Flatten, Dense, Dropout
from keras.api.utils import to_categorical
from keras.api.initializers import Orthogonal
from sklearn.model_selection import train_test_split
from keras.api.callbacks import ModelCheckpoint, ReduceLROnPlateau
import numpy as np
import matplotlib.pyplot as plt


actions = ["hello", "thanks", "sorry", "hate", "hungry",
           "sick", "tired", "mind", "person", "think",
           "friend", "school", "police", "rice", "bed"]
dataset_path = 'dataset/'
data = np.concatenate([np.load(dataset_path + f'seq_{action}.npy') for action in actions], axis=0)

x_data = data[:, :, :-1]
labels = data[:, 0, -1]
y_data = to_categorical(labels, num_classes=len(actions))
x_data = x_data.astype(np.float32)
y_data = y_data.astype(np.float32)

x_train, x_val, y_train, y_val = train_test_split(x_data, y_data, test_size=0.2, random_state=2024)

init = Orthogonal(gain=1.0, seed=None)
dr = 0.3

model = Sequential()
model.add(Input(x_train.shape[1:]))

model.add(LSTM(256, return_sequences=True, kernel_initializer=init))
model.add(Dropout(dr))
model.add(LSTM(128, return_sequences=True, kernel_initializer=init))
model.add(Flatten())

model.add(Dense(64, activation='relu', kernel_initializer=init))
model.add(Dropout(dr))
model.add(Dense(len(actions), activation='softmax', kernel_initializer=init))

model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit(
    x_data,
    y_data,
    validation_data=(x_val, y_val),
    epochs=100,
    batch_size=32,
    callbacks=[
        ModelCheckpoint('models/test_LSTM.keras', verbose=2, save_best_only=True, mode='auto'),
        ReduceLROnPlateau(factor=0.5, patience=50, verbose=2, mode='auto')
    ]
)

fig, loss_ax = plt.subplots(figsize=(12, 5))
acc_ax = loss_ax.twinx()

loss_ax.plot(history.history['loss'], 'y', label='train_loss')
loss_ax.plot(history.history['val_loss'], 'r', label='val_loss')
loss_ax.set_xlabel('epoch')
loss_ax.set_ylabel('loss')
loss_ax.legend(loc='upper right')

acc_ax.plot(history.history['accuracy'], 'b', label='train_acc')
acc_ax.plot(history.history['val_accuracy'], 'g', label='val_acc')
acc_ax.set_ylabel('accuracy')
acc_ax.legend(loc='center right')
plt.title('LSTM')
plt.ylim(0.0, 1.1)
plt.show()
