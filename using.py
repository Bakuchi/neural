from keras.models import load_model
from keras.preprocessing import sequence
import numpy as np
import pymorphy2
import re

dictionary = np.load("dictionary.npy").tolist()

model = load_model('m.h5')

words="Короч, все должно работать, хехе"

def text_handler(text):
    morph = pymorphy2.MorphAnalyzer()
    text = text.lower()
    alph = 'абвгдеёжзийклмнопрстуфхцчшщъыьэюя'
    
    cleaned_text = ''
    for char in text:
        if (char.isalpha() and char[0] in alph) or (char == ' '):
            cleaned_text += char
        
    result = []
    for word in cleaned_text.split():
        result.append(morph.parse(word)[0].normal_form)
    row = []
    for word in result:
        if word in dictionary:
           row.append(dictionary.index(word) + 1)
        else:
           row.append(0)             
    return [row]

X = sequence.pad_sequences(np.array(text_handler(words)), maxlen=100)
result = model.predict(X)
print(result)
