import os
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

DATA_PATH = "data"
WORDS = ["goodbye", "hello", "no", "please", "yes", "thanks", "sorry"]
SEQUENCE_LENGTH = 10

X, y = [], []

for idx, word in enumerate(WORDS):
    word_path = os.path.join(DATA_PATH, word)
    for file in os.listdir(word_path):
        data = np.load(os.path.join(word_path, file))
        X.append(data)
        y.append(idx)

X = np.array(X)
y = to_categorical(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = Sequential([
    LSTM(64, return_sequences=True, activation='relu', input_shape=(SEQUENCE_LENGTH, X.shape[2])),
    LSTM(64, activation='relu'),
    Dense(64, activation='relu'),
    Dense(len(WORDS), activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

model.fit(X_train, y_train, epochs=30, validation_data=(X_test, y_test))

model.save("sign_language_model.h5")
