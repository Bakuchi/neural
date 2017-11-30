import numpy as np

X_train = np.load("X_train.npy")
X_test = np.load("X_test.npy")
y_train = np.load("y_train.npy")
y_test = np.load("y_test.npy")
dict_len = 100000

model = Sequential()
model.add(Embedding(dictionary_length + 1, 128, input_length=100))
model.add(LSTM(64, return_sequences=True))
model.add(LSTM(64))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train, y_train, validation_data=(X_test, y_test), batch_size=32, epochs=1)

scores = model.evaluate(X_test, y_test, verbose=0)
model.save("m.h5")
print("Accuracy: %.2f%%" % (scores[1]*100))


