from keras.utils import normalize
from keras.models import load_model
import pickle
from termcolor import *
import colorama

colorama.init()

dictionary = pickle.load(open('model_dictionary', 'rb'))
model = load_model('classifier_dl')

while True:

	text = input('Enter text:\n').split(' ')

	if text[0] == 'exit':
		break

	feature_set = []

	for entry in dictionary:
		feature_set.append(text.count(entry[0]))

	feature_set = normalize(feature_set)
	result = model.predict([feature_set])
	print(f'Result: {result}')

	if result[0] > 0.5:
		cprint('Spam', color = 'red')
	else:
		cprint('Not spam', color = 'green')
