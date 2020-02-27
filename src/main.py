# *coderTazz
# main file to run all experiments
# experiment type and number of train/test samples passed as arguments


# Imports
import os
import argparse
import math
from PIL import Image
from math import *
import cv2
from tqdm import tqdm
import random
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
import scipy.special
import scipy.stats
import matplotlib.pyplot as plt

# CONFIGURE
TRAIN_POS_PATH = '../../savedPics/train/pos/'
TRAIN_NEG_PATH = '../../savedPics/train/neg/'
TEST_POS_PATH = '../../savedPics/test/pos/'
TEST_NEG_PATH = '../../savedPics/test/neg/'
posSTR = 'pos'
workWithDiagonalCov = True
addNoise = True
negSTR = 'neg'
dim = 0
extensionSTR = ".jpg"
mixture_model_no = 2
fcta_factor_no = 5
normalizeScale = 0
rocPoints = 20
trSize = 0
teSize = 0
res = 0


# Method to randomly collect required number of train/test samples
def getData(isTrain):
	
	global trSize
	global teSize

	if(isTrain):
		target = trSize
		posRange = len([name for name in os.listdir(TRAIN_POS_PATH)])
		negRange = len([name for name in os.listdir(TRAIN_NEG_PATH)])
		posPath = TRAIN_POS_PATH
		negPath = TRAIN_NEG_PATH
		bias = 2
		print('Getting Training Data..', end = "")
	else:
		target = teSize
		posRange = len([name for name in os.listdir(TEST_POS_PATH)])
		negRange = len([name for name in os.listdir(TEST_NEG_PATH)])
		posPath = TEST_POS_PATH
		negPath = TEST_NEG_PATH
		bias = 2
		print('Getting Testing Data..', end = "")

	
	data = []
	label = []
	dice = 0
	trno = 0
	i = 0
	seenPos = {}
	seenNeg = {}

	while(i < target):

		if(dice%bias != 0):
			seenFlag = True
			while(seenFlag):
				imageIndex = random.randint(0, posRange)
				if(not (imageIndex in seenPos)):
					img = cv2.imread(posPath + posSTR + str(imageIndex) + extensionSTR)
					# print(posPath + posSTR + str(imageIndex) + extensionSTR)
					if(img is None):
						continue
					data.append(img)
					label.append(1)
					seenFlag = False
					seenPos[imageIndex] = 0
					dice += 1
					i += 1
		else:
			seenFlag = True
			while(seenFlag):
				imageIndex = random.randint(0, negRange)
				if(not (imageIndex in seenNeg)):
					img = cv2.imread(negPath + negSTR + str(imageIndex) + extensionSTR)
					# print(negPath + negSTR + str(imageIndex) + extensionSTR)
					if(img is None):
						continue
					data.append(img)
					label.append(0)
					seenFlag = False
					seenNeg[imageIndex] = 0
					dice += 1
					i += 1
	print('...Done')

	return data, label

# Returns train test split
def getTrainTestData():
	return getData(True), getData(False)

# Stretching each (DxDx3) image to (D*D*3) column vector
def stretchX(data):
	global dim
	global res
	dim = data[0].shape[0] * data[0].shape[1] * data[0].shape[2]
	res = data[0].shape[0]
	for i in range(len(data)):
		data[i] = data[i].reshape(dim)
	return data

def getPosNegSplit(data, label):
	pos = []
	neg = []
	posLabel = []
	negLabel = []
	for i in range(len(label)):
		if(label[i] == 1):
			pos.append(data[i])
			posLabel.append(1)
		else:
			neg.append(data[i])
			negLabel.append(0)

	return np.array(pos), np.array(posLabel), np.array(neg), np.array(negLabel)

def getBernoulliPrior(trainLabel):

	classCount = np.unique(trainLabel).shape[0]
	classes = classCount*[0]

	for i in range(len(trainLabel)):
		classes[trainLabel[i]] += 1
	return classes/np.sum(classes)

def getMuAndCov(m):
	global workWithDiagonalCov

	# Mean is sum(x)/trSize
	# Variance is (x - mu).T*(x - mu)
	mu = np.sum(m, axis = 0)/m.shape[0]
	m_mu = m - mu
	cov = np.dot((m_mu).T, (m_mu))/(m.shape[0])
	if(workWithDiagonalCov):
		cov = getDiagCov(cov)
	return np.array(mu), np.array(cov)

def plotMuAndCov(mu, cov):
	global dim
	imgMu = mu.reshape(res, res, 3)
	imgMu = (imgMu - imgMu.min())/(imgMu.max() - imgMu.min())*255
	img = Image.fromarray(imgMu, 'RGB')
	print('Mean')
	img.show()

	# covImg = []
	# for i in range(len(cov)):
	# 	covImg.append(cov[i][i])
	# covImg = np.array(covImg)
	# imgCov = covImg.reshape(res, res, 3)
	# imgCov = (imgCov - imgCov.min())/(imgCov.max() - imgCov.min())*255
	# imgCov = np.sqrt(imgCov)
	# img = Image.fromarray(imgCov, 'RGB')
	# print('Covariance')
	# img.show()

def getDiagCov(cov):
	arr = [[0 for i in range(len(cov))] for j in range(len(cov))]
	for i in range(len(cov)):
		arr[i][i] = cov[i][i]
	return np.array(arr)

def getMultiGuassianProbabilityDensity(var, m_mu, var_sqrt_det, var_inv):
	cnst_term = (1/(2*math.pi))**(1/2)
	return cnst_term*(1/var_sqrt_det)*math.e**((-.5)*(m_mu.dot(var_inv).dot(m_mu.T)))

def getLogMultiGuassianProbabilityDensity(var, m_mu, var_sqrt_det, var_inv):
	frst_term = -(var.shape[0]/2)*math.log(2*math.pi)
	# print(var_sqrt_det)
	scnd_term = -math.log(var_sqrt_det)
	thrd_term = (-.5)*(m_mu.dot(var_inv).dot(m_mu.T))
	return frst_term + scnd_term + thrd_term

def getLogMultiTProbabilityDensity(var, m_mu, var_sqrt_det, var_inv, v):
	val = 0
	# val += math.log(math.gamma((v + dim)/2)), will remain constant
	# val -= (dim/2)*math.log(v*math.pi), will remain constant
	val -= math.log(var_sqrt_det)
	val -= math.log(math.gamma((v)/2))
	val += -((v + dim)/2)*math.log(1 + (m_mu.dot(var_inv).dot(m_mu.T))/v)
	return val

def addNoise(m):
	m = [d + random.uniform(0, 0.5) for d in m]
	return np.array(m)

def normalize(tr_m, te_m):
	df_tr = pd.DataFrame(tr_m)
	df_te = pd.DataFrame(te_m)
	return np.array((df_tr - df_tr.min())/(normalizeScale)), np.array((df_te - df_tr.min())/(normalizeScale))

def performEM(model, data):
	global dim
	global mixture_model_no
	
	noSamples = data.shape[0]
	valRange = np.amax(data)
	# print(data[0].shape)

	if(model == 'gmm'):
		
		def lower_bound_update():
			r = []
			# print(cov_sqrt_det[0], cov_sqrt_det[1])
			# print('This')
			for sample in data:
				ri = []
				denom = 0
				for i in range(mixture_model_no):
					numer = other_var[i]*getLogMultiGuassianProbabilityDensity(cov[i], (sample - mu[i]).reshape(1, dim), cov_sqrt_det[i], cov_inv[i])
					denom += numer
					ri.append(numer)
				ri /= denom
				r.append(ri)
			r = np.array(r).reshape(noSamples, mixture_model_no)
			r = r.T
			# print(r.shape)
			return r, None

		def other_var_update(r, nil):
			return (np.sum(r, axis = 1)/np.sum(np.sum(r, axis = 1))).reshape(mixture_model_no, 1)

		def mu_update(r):
			return r.dot(data)/np.sum(r, axis = 1).reshape(mixture_model_no, 1)

		def cov_update(r, mu):
			for i in range(mixture_model_no):
				# print(mu[i].shape)
				rk = r[i].reshape(noSamples, 1)
				cov[i] = np.dot((rk*(data - (mu[i]).reshape(1, dim))).T, (data - (mu[i]).reshape(1, dim)))/np.sum(rk)
				if(workWithDiagonalCov):
					cov[i] = getDiagCov(cov[i])
			return cov

		def stoppingCriterionMet(mu, prevMu):
			return np.sum(abs(mu - prevMu) < 0.00001) == dim*mixture_model_no

		other_var = np.random.dirichlet(np.ones(mixture_model_no))
		mu = [[0 for _ in range(dim)] for _ in range(mixture_model_no)]
		for i in range(mixture_model_no):
			for j in range(dim):
				mu[i][j] = random.uniform(0, valRange)
		mu = np.array(mu)
		cov = []
		for i in range(mixture_model_no):
			tmp = [[0 for _ in range(dim)] for _ in range(dim)]
			for j in range(dim):
				tmp[j][j] = random.uniform(0, valRange)
			cov.append(tmp)
		cov = np.array(cov)
		# print(cov, np.linalg.det(cov))
		cov_sqrt_det = np.sqrt(np.linalg.det(cov))
		cov_inv = np.linalg.inv(cov)

	if(model == 'fcta'):

		def lower_bound_update():
			eh = []
			ehh = []
			frst_term = np.linalg.inv((other_var.T.dot(cov_inv).dot(other_var) + np.identity(fcta_factor_no)))
			scnd_term = (other_var.T).dot(cov_inv)
			for i, sample in enumerate(data):
				eh.append(frst_term.dot(scnd_term.dot((sample - mu).reshape(dim, 1))))
				ehh.append(frst_term + eh[i].dot(eh[i].T))
			return np.array(eh).reshape(noSamples, fcta_factor_no, 1), np.array(ehh).reshape(noSamples, fcta_factor_no, fcta_factor_no)

		def other_var_update(eh, ehh):
			frst_term = np.zeros((dim, fcta_factor_no))
			scnd_term = np.zeros((fcta_factor_no, fcta_factor_no))
			for i, sample in enumerate(data):
				frst_term += (sample - mu).reshape(dim, 1).dot(eh[i].T)
				scnd_term += ehh[i]
			return frst_term.dot(np.linalg.inv(scnd_term))

		def mu_update(nil):
			# print('Mu done')
			return np.sum(data, axis = 0)/noSamples

		def cov_update(eh, mu):
			cov = np.zeros((dim, dim))
			for i, sample in enumerate(data):
				cov += (sample - mu).reshape(dim, 1).dot((sample - mu).reshape(1, dim)) - (other_var.dot(eh[i])).dot((sample - mu).reshape(1, dim))
			cov /= noSamples
			if(workWithDiagonalCov):
				cov = getDiagCov(cov)
			# print('Cov Done')
			return cov

		def stoppingCriterionMet(mu, prevMu):
			return np.sum(abs(mu - prevMu) < 0.00001) == dim

		other_var = np.array([[0 for _ in range(fcta_factor_no)] for _ in range(dim)])
		for i in range(dim):
			for j in range(fcta_factor_no):
				other_var[i][j] = random.uniform(0, 1)
		mu = []
		for i in range(dim):
			mu.append(random.uniform(0, 1))
		mu = np.array(mu)
		cov = [[0 for _ in range(dim)] for _ in range(dim)]
		for j in range(dim):
			cov[j][j] = random.uniform(0, 1)
		cov = np.array(cov)
		# print(np.linalg.det(cov))
		cov_sqrt_det = np.sqrt(np.linalg.det(cov))
		cov_inv = np.linalg.inv(cov)

	if(model == 'tdst'):

		def lower_bound_update():
			eh = []
			elogh = []
			for sample in data:
				eh.append((other_var + dim)/(other_var + ((sample - mu).reshape(1, dim)).dot(cov_inv).dot((sample - mu).reshape(dim, 1))))
				elogh.append(scipy.special.digamma((other_var + dim)/2) - math.log((other_var + ((sample - mu).reshape(1, dim)).dot(cov_inv).dot((sample - mu).reshape(dim, 1)))/2))
			return np.array(eh).reshape(noSamples, 1), np.array(elogh).reshape(noSamples, 1)

		def other_var_update(eh, elogh):
			v = 3
			frst_term = (dim*elogh - dim*math.log(2*math.pi) - math.log(cov_sqrt_det) - (data - (mu).reshape(1, dim)).dot(cov_inv).dot((data - (mu).reshape(1, dim)).T).dot(eh))/2
			frst_term = np.sum(frst_term)
			prevObjective = -1*float('inf')
			while(True):
				scnd_term = (v/2)*math.log(v/2) - math.log(math.gamma(v/2)) + (v/2 - 1)*elogh - (v/2)*eh
				scnd_term = np.sum(scnd_term)
				objective = frst_term + scnd_term
				# print(v, objective)
				if(objective > prevObjective):
					v += 1
					prevObjective = objective
					continue
				else:
					return v - 1

		def mu_update(eh):
			return np.sum(eh*(data), axis = 0)/np.sum(eh)

		def cov_update(eh, mu):
			cov = np.dot((eh*(data - mu)).T, (data - mu))/np.sum(eh)
			if(workWithDiagonalCov):
				cov = getDiagCov(cov)
			return cov

		def stoppingCriterionMet(mu, prevMu):
			return np.sum(abs(mu - prevMu) < 0.00001) == dim

		# v = random.randint(1, 100000)
		other_var = random.randint(3, 100000)
		mu = []
		for i in range(dim):
			mu.append(random.uniform(0, 1))
		mu = np.array(mu)
		cov = [[0 for _ in range(dim)] for _ in range(dim)]
		for j in range(dim):
			cov[j][j] = random.uniform(0, 1)
		cov = np.array(cov)
		cov_sqrt_det = np.sqrt(np.linalg.det(cov))
		cov_inv = np.linalg.inv(cov)



	# Main EM Loop
	eStep = True
	em = 0
	while(True):
		if(eStep):
			# E-Step
			# print('E-Step')
			q1, q2 = lower_bound_update()
			eStep = False
			# print('E-Step Done')
		else:
			# M-Step
			if(model == 'gmm'):
				other_var = other_var_update(q1, q2)
			# print(other_var.shape)
			mu = mu_update(q1)
			if(model == 'fcta'):
				other_var = other_var_update(q1, q2)
			cov = cov_update(q1, mu)
			# print(np.linalg.det(cov))
			cov_sqrt_det = np.sqrt(np.linalg.det(cov))
			# print(cov_sqrt_det)
			cov_inv = np.linalg.inv(cov)
			if(model == 'tdst'):
				other_var = other_var_update(q1, q2)
			eStep = True
			if(em != 1):
				if(stoppingCriterionMet(mu, prevMu)):
					break
			prevMu = np.copy(mu)

		em += 1
		# print(em)

	return other_var, mu, cov

def ROC(label, pred):

	tpr, fpr = getMetrics(label, pred)

	# plt.scatter(fpr, tpr)
	plt.plot(tpr, tpr, color = 'black', label = 'Random Guessing')
	plt.plot(fpr, tpr, color = 'green', label = 'My Model')
	plt.title('ROC')
	plt.legend(loc='upper center')
	plt.show()

def getMetrics(label, pred):
	P = sum(label)
	N = len(label) - P

	tpr, fpr = [], []
	mn = min(pred)
	mx = max(pred)
	diff = mx - mn
	thresholds = np.arange(mn, mx + diff/rocPoints, diff/rocPoints)

	for t in thresholds:
		tp, fp = 0, 0
		for i in range(len(pred)):
			if(pred[i] > t):
				if(label[i] == 1):
					tp += 1
				else:
					fp += 1
		tpr.append(tp/(P))
		fpr.append(fp/(N))

	return tpr, fpr

def fctaModel(trainData, trainLabel, testData, testLabel):
	global normalizeScale
	normalizeScale = 60

	trainData, testData = normalize(trainData, testData)

	trPos, trPosLabel, trNeg, trNegLabel = getPosNegSplit(trainData, trainLabel)

	# if(addNoise):
	# 	trPos = addNoise(trPos)
	# 	trNeg = addNoise(trNeg)

	print('Fitting Parameters for Face Class')
	psiPos, muPos, covPos = performEM('fcta', trPos)
	covPos_sqrt = np.sqrt(np.linalg.det(covPos))
	covPos_inv = np.linalg.inv(covPos)
	print('Done')
	print('Fitting Parameters for Non-Face Class')
	psiNeg, muNeg, covNeg = performEM('fcta', trNeg)
	covNeg_sqrt = np.sqrt(np.linalg.det(covNeg))
	covNeg_inv = np.linalg.inv(covNeg)
	print('Done')

	# Get Bernoulli Prior
	classPriors = getBernoulliPrior(trainLabel)

	prediction, rocPoints = [], []
	pbar = tqdm(range(1, len(testData) + 1), unit = 'Test Images')
	for imgInd in pbar:
		img = testData[imgInd - 1]
		posXW = getLogMultiGuassianProbabilityDensity(covPos + psiPos.dot(psiPos.T), img - muPos, covPos_sqrt, covPos_inv)
		negXW = getLogMultiGuassianProbabilityDensity(covNeg + psiNeg.dot(psiNeg.T), img - muNeg, covNeg_sqrt, covNeg_inv)
		# denom = posXW + negXW
		posProb = posXW*classPriors[1]
		negProb = negXW*classPriors[0]
		rocPoints.append(posProb)
		if(posProb > negProb):
			# print('face')
			prediction.append(1)
		else:
			# print('non-face')
			prediction.append(0)

	print('Accuracy: ' + str(np.mean(np.array(prediction) == np.array(testLabel))*100) + '%')
	print()
	print('Confusion Matrix:')
	print(confusion_matrix(prediction, testLabel))


	ROC(testLabel, rocPoints)


	# plotMuAndCov(muPos, covPos)
	# plotMuAndCov(muNeg, covNeg)

def tdistModel(trainData, trainLabel, testData, testLabel):
	global normalizeScale
	normalizeScale = 50

	trainData, testData = normalize(trainData, testData)

	trPos, trPosLabel, trNeg, trNegLabel = getPosNegSplit(trainData, trainLabel)

	# if(addNoise):
	# 	trPos = addNoise(trPos)
	# 	trNeg = addNoise(trNeg)

	print('Fitting Parameters for Face Class')
	vPos, muPos, covPos = performEM('tdst', trPos)
	covPos_sqrt = np.sqrt(np.linalg.det(covPos))
	covPos_inv = np.linalg.inv(covPos)
	print('Done')
	print('Fitting Parameters for Non-Face Class')
	vNeg, muNeg, covNeg = performEM('tdst', trNeg)
	covNeg_sqrt = np.sqrt(np.linalg.det(covNeg))
	covNeg_inv = np.linalg.inv(covNeg)
	print('Done')

	# Get Bernoulli Prior
	classPriors = getBernoulliPrior(trainLabel)

	prediction, rocPoints = [], []
	pbar = tqdm(range(1, len(testData) + 1), unit = 'Test Images')
	for imgInd in pbar:
		img = testData[imgInd - 1]
		posXW = getLogMultiTProbabilityDensity(covPos, (img - muPos).reshape(1, dim), covPos_sqrt, covPos_inv, vPos)
		negXW = getLogMultiTProbabilityDensity(covNeg, (img - muNeg).reshape(1, dim), covNeg_sqrt, covNeg_inv, vNeg)
		# denom = posXW + negXW
		posProb = posXW*classPriors[1]
		negProb = negXW*classPriors[0]
		rocPoints.append(posProb)
		# print(posProb/(negProb + posProb), negProb/(negProb + posProb))
		if(posProb > negProb):
			# print('face')
			prediction.append(1)
		else:
			# print('non-face')
			prediction.append(0)

	print('Accuracy: ' + str(np.mean(np.array(prediction) == np.array(testLabel))*100) + '%')
	print()
	print('Confusion Matrix:')
	print(confusion_matrix(prediction, testLabel))

	ROC(testLabel, rocPoints)


	# plotMuAndCov(muPos, covPos)
	# plotMuAndCov(muNeg, covNeg)

def gaussianMixtureModel(trainData, trainLabel, testData, testLabel):
	global normalizeScale
	normalizeScale = 60

	trainData, testData = normalize(trainData, testData)

	trPos, trPosLabel, trNeg, trNegLabel = getPosNegSplit(trainData, trainLabel)

	# if(addNoise):
	# 	trPos = addNoise(trPos)
	# 	trNeg = addNoise(trNeg)

	print('Fitting Parameters for Face Class')
	lmbdaPos, muPos, covPos = performEM('gmm', trPos)
	covPos_sqrt = np.sqrt(np.linalg.det(covPos))
	covPos_inv = np.linalg.inv(covPos)
	print('Done')
	print('Fitting Parameters for Non-Face Class')
	lmbdaNeg, muNeg, covNeg = performEM('gmm', trNeg)
	covNeg_sqrt = np.sqrt(np.linalg.det(covNeg))
	covNeg_inv = np.linalg.inv(covNeg)
	print('Done')

	# Get Bernoulli Prior
	classPriors = getBernoulliPrior(trainLabel)

	# Sanity Check
	if(np.array_equal(muPos[0], muPos[1])):
		print('Broken')
		return

	prediction, rocPoints = [], []
	pbar = tqdm(range(1, len(testData) + 1), unit = 'Test Images')
	for imgInd in pbar:
		img = testData[imgInd - 1]
		posSum = 0
		negSum = 0
		for i in range(mixture_model_no):
			posXW = getLogMultiGuassianProbabilityDensity(covPos[i], (img - muPos[i]).reshape(1, dim), covPos_sqrt[i], covPos_inv[i])
			negXW = getLogMultiGuassianProbabilityDensity(covNeg[i], (img - muNeg[i]).reshape(1, dim), covNeg_sqrt[i], covNeg_inv[i])
			posSum += lmbdaPos[i]*posXW
			negSum += lmbdaNeg[i]*negXW
		# denom = posXW + negXW
		posProb = posSum*classPriors[1]
		negProb = negSum*classPriors[0]
		rocPoints.append(posProb)
		if(posProb > negProb):
			# print('face')
			prediction.append(1)
		else:
			# print('non-face')
			prediction.append(0)

	print('Accuracy: ' + str(np.mean(np.array(prediction) == np.array(testLabel))*100) + '%')
	print()
	print('Confusion Matrix:')
	print(confusion_matrix(prediction, testLabel))

	ROC(testLabel, rocPoints)

	# for i in range(mixture_model_no):
	# 	plotMuAndCov(muNeg[i], covNeg[i])
	# 	plotMuAndCov(muPos[i], covPos[i])

def gaussianMultivariate(trainData, trainLabel, testData, testLabel):
	global normalizeScale
	normalizeScale = 55

	trainData, testData = normalize(trainData, testData)

	trPos, trPosLabel, trNeg, trNegLabel = getPosNegSplit(trainData, trainLabel)
	# print(trPos.shape)
	# if(addNoise):
	# 	trPos = addNoise(trPos)
	# 	trNeg = addNoise(trNeg)

	# Fitting Parameters
	# Positive Class
	muTrainPos, varTrainPos = getMuAndCov(trPos)
	varTrainPos_sqrt = np.sqrt(np.linalg.det(varTrainPos))
	varTrainPos_inv = np.linalg.inv(varTrainPos)
	# Negative Class
	muTrainNeg, varTrainNeg = getMuAndCov(trNeg)
	varTrainNeg_sqrt = np.sqrt(np.linalg.det(varTrainNeg))
	varTrainNeg_inv = np.linalg.inv(varTrainNeg)

	# Get Bernoulli Prior
	classPriors = getBernoulliPrior(trainLabel)

	prediction, rocPoints = [], []
	pbar = tqdm(range(1, len(testData) + 1), unit = 'Test Images')
	for imgInd in pbar:
		img = testData[imgInd - 1]
		# print((img - muTrainPos).reshape(1, dim).shape)
		posXW = getLogMultiGuassianProbabilityDensity(varTrainPos, (img - muTrainPos).reshape(1, dim), varTrainPos_sqrt, varTrainPos_inv)
		negXW = getLogMultiGuassianProbabilityDensity(varTrainNeg, (img - muTrainNeg).reshape(1, dim), varTrainNeg_sqrt, varTrainNeg_inv)
		# denom = posXW + negXW
		posProb = posXW*classPriors[1]
		negProb = negXW*classPriors[0]
		rocPoints.append(posProb)
		# print(posProb/(posXW + negXW), negProb/(posXW + negXW))
		if(posProb > negProb):
			# print('face')
			prediction.append(1)
		else:
			# print('non-face')
			prediction.append(0)

	print('Accuracy: ' + str(np.mean(np.array(prediction) == np.array(testLabel))*100) + '%')
	print()
	print('Confusion Matrix:')
	print(confusion_matrix(prediction, testLabel))

	ROC(testLabel, rocPoints)
	# plotMuAndCov(muTrainPos, varTrainPos)
	# plotMuAndCov(muTrainNeg, varTrainNeg)

if __name__=="__main__":

	parser = argparse.ArgumentParser()
	parser.add_argument("--tr_sz", default=2000, type=int)
	parser.add_argument("--te_sz", default=100, type=int)
	parser.add_argument("--model_type", default="gaus", type=str, help="gaus | gmm | tdst | fcta")

	args = parser.parse_args()

	trSize = args.tr_sz
	teSize = args.te_sz

	(trainData, trainLabel), (testData, testLabel) = getTrainTestData()
	trainData = stretchX(trainData)
	testData = stretchX(testData)

	print()

	if(args.model_type == 'gaus'):
		print('Running Multivariate Gaussian')
		print()
		gaussianMultivariate(trainData, trainLabel, testData, testLabel)
	elif(args.model_type == 'gmm'):
		print('Running Mixture of Multivariate Gaussians')
		print()
		gaussianMixtureModel(trainData, trainLabel, testData, testLabel)
	elif(args.model_type == 'tdst'):
		print('Running Multivariate T-Dist')
		print()
		tdistModel(trainData, trainLabel, testData, testLabel)
	elif(args.model_type == 'fcta'):
		print('Running Multivariate Factor Analysis')
		print()
		fctaModel(trainData, trainLabel, testData, testLabel)

	# print('Done')


