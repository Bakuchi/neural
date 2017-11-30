from sklearn.cross_validation import train_test_split
from keras.preprocessing import sequence
import pandas as pd
import numpy as np
import re

dataset = pd.read_csv('cleaned_data.csv', index_col=0).dropna()

X = []
y = []
dictionary = []

dataset['label'].replace(-1, 0, inplace=True)
dataset = dataset.sample(frac=1)
tpls = list(dataset.itertuples())

for tpl in tpls:
    row = []
    words = re.sub("[^\w]", " ",  tpl[1]).split()
    for word in words:
        if word not in dictionary:
            dictionary.append(word)
        row.append(dictionary.index(word) + 1)
    X.append(row)
    y.append(tpl[2])

X = sequence.pad_sequences(np.array(X), maxlen=100)
y = np.array(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

np.save("X_train", X_train)
np.save("X_test", X_test)
np.save("y_train", y_train)
np.save("y_test", y_test)
np.save("dictionary", dictionary)
