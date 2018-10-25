import os
from collections import Counter
import pickle

def make_dictionary():

	directory = 'emails/'
	files = os.listdir(directory)
	emails = [directory + email for email in files]

	words = []
	for email in emails:
		with open(email, encoding = 'latin-1') as file:
			blob = file.read()
			words += blob.split(' ')
			print('Making Dictionary')

	for i in range(len(words)):
		if not words[i].isalpha():
			words[i] = ''

	dictionary = Counter(words)
	del dictionary['']

	return dictionary.most_common(10000)

def make_dataset(dictionary):

	directory = 'emails/'
	files = os.listdir(directory)
	emails = [directory + email for email in files]

	features = []
	labels = []

	c = len(emails)
	for email in emails:
		feature_vector = []

		with open(email, encoding = 'latin-1') as file:
			words = file.read().split(' ')

		for entry in dictionary:
			feature_vector.append(words.count(entry[0]))
		features.append(feature_vector)

		if 'ham' in email:
			labels.append(0)

		else:
			labels.append(1)

		print(f'Making dataset: {c}')
		c -= 1

	return features, labels

d = make_dictionary()

filename = 'model_dictionary'
pickle.dump(d, open(filename, 'wb'))

features, labels = make_dataset(d)
pickle.dump(features, open('features', 'wb'))
pickle.dump(labels, open('labels', 'wb'))
