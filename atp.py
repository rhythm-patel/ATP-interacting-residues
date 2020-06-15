import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from imblearn.ensemble import BalancedBaggingClassifier
import csv
import sys
import datetime

now = datetime.datetime.now()
print ("Current date and time : ")
print (now.strftime("%Y-%m-%d %H:%M:%S"))

def buildTrain(chunkSize, amino):
	train = pd.read_csv('train.data')
	x_train_seq = train.iloc[:, 1]
	X_train = []
	y_train = []
	xBuffer = ""
	for yo in range((chunkSize-1)//2):
		xBuffer += "X"

	for seq in x_train_seq:				# one particular sequence
		seq = xBuffer + seq + xBuffer
		lenSeq = len(seq)
		seqVector = []
		labelList = []
		c = 0

		for i in range(0, lenSeq-(chunkSize-1)):		# one 7 sized chunk

			chunk = seq[i:i+chunkSize]
			mid = chunk[(chunkSize-1)//2]

			if (mid.islower()):			# interacting residue
				label = 1				# -1/1 for each chunk
			else:						# non-interacting residue
				label = -1				# -1/1 for each chunk
			labelList.append(label)		# [-1,-1,1] for each sequence

			seqTemp = []
			for letter in chunk:
				position = amino[letter.upper()]
				temp = [0]*21
				temp[position] = 1
				seqTemp += temp 		# [0,0,0,1,0,0,1,0] for each chunk

			# [[0,0,0,1,0,0,1,0, 0,1,0,0,1,0,0,1], [1,0,0,0,1,0,0,1, 0,0,0,1,0,1,0,1]] for full dataset
			X_train.append(seqTemp)

		y_train += labelList

	X_train = np.asarray(X_train)
	y_train = np.asarray(y_train)

	return X_train, y_train, xBuffer


def trainModel(X_train, y_train, modelChoice, balancedOption):

	if (modelChoice == 1):
		model = SVC(kernel="rbf")  # our ML model
	else:
		model = RandomForestClassifier(n_estimators=1000)

	balModel = model

	if (balancedOption.lower() == 'y'):
		balModel = BalancedBaggingClassifier(
			model, random_state=0, n_estimators=51)
		balModel.fit(X_train, y_train)  # fit the model by x & y of train
		print("Model trained!")
		return balModel
	else:
		print("Model trained!")
		return model


def buildTest(chunkSize, xBuffer, amino):
	test = pd.read_csv('test1.txt')
	x_test_seq = test.iloc[:, 1]
	testSeq = ""

	for i in x_test_seq:
		testSeq += i

	testSeq = xBuffer + testSeq + xBuffer
	lenTestSeq = len(testSeq)
	X_test = []

	for i in range(0, lenTestSeq-(chunkSize-1)):		# one 7 sized chunk

		testSeqTemp = []
		chunk = testSeq[i:i+chunkSize]
		mid = chunk[(chunkSize-1)//2]

		for letter in chunk:
			position = amino[letter.upper()]
			temp = [0]*21
			temp[position] = 1
			testSeqTemp += temp 		# [0,0,0,1,0,0,1,0] for each chunk

		# [0,0,0,1,0,0,1,0, 0,1,0,0,1,0,0,1] for each sequence
		X_test.append(testSeqTemp)

	X_test = np.asarray(X_test)

	return X_test


def predict(X_test, model):
	y_test = model.predict(X_test)  # predict the values from the model
	return y_test


def save(outputFile, y_test):
	ID = 10001
	output = [["ID", "Lable"]]

	for i in range(len(y_test)):
		temp = []
		temp.append(ID)  # adds the IDs
		temp.append(int(y_test[i]))  # adds the predicted y values i.e 1/-1
		output.append(temp)
		ID += 1

	with open(outputFile, 'w', newline='') as file:
		writer = csv.writer(file)
		writer.writerows(output)

	print("Output exported in", outputFile)


def inputStuff():
	print("Enter 'd' for default settings for the following")
	modelChoice = input(
		"Enter 1 for SVC, Enter 2 for Random Forest Classifier: ")
	if (modelChoice.lower() == 'd'):
		modelChoice = 1
	elif (int(modelChoice) not in [1, 2]):
		print("Invalid Input")
		sys.exit(0)

	chunkSize = input("Enter window size: ")

	if (chunkSize.lower() == 'd'):
		chunkSize = 13
	elif (int(chunkSize) % 2 == 0 or int(chunkSize) == 1):
		print("Window size should be odd & greater than 1")
		sys.exit(0)

	balancedOption = input("Enter Y/N to balance training data: ")
	if (balancedOption.lower() == 'd'):
		balancedOption = 'y'
	elif (balancedOption not in ['Y', 'y', 'N', 'n']):
		print("Invalid Input")
		sys.exit(0)

	outputFile = input("Enter output file name: ")

	if (outputFile.lower() == 'd'):
		outputFile = 'output.csv'

	return int(modelChoice), int(chunkSize), balancedOption, outputFile


if __name__ == "__main__":
	amino = {'A': 0, 'C': 1, 'D': 2, 'E': 3, 'F': 4, 'G': 5, 'H': 6, 'I': 7, 'K': 8, 'L': 9, 'M': 10,
			 'N': 11, 'P': 12, 'Q': 13, 'R': 14, 'S': 15, 'T': 16, 'V': 17, 'W': 18, 'Y': 19, 'X': 20}
	modelChoice, chunkSize, balancedOption, outputFile = inputStuff()
	X_train, y_train, xBuffer = buildTrain(chunkSize, amino)
	model = trainModel(X_train, y_train, modelChoice, balancedOption)
	X_test = buildTest(chunkSize, xBuffer, amino)
	y_test = predict(X_test, model)
	save(outputFile, y_test)

	now = datetime.datetime.now()
	print ("Current date and time : ")
	print (now.strftime("%Y-%m-%d %H:%M:%S"))
