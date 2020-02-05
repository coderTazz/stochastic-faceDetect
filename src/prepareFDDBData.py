# *coderTazz
# python script to create n train and m test samples from FDDB for this repo
# number of samples required can be passed as arguments


# Imports
import os
import math
from PIL import Image
from math import *
from os import walk
from tqdm import tqdm
import random



# CONFIGURE
ANNOTATION_PATH = '../../FDDB-folds/'
FILE_PREFIX = 'FDDB-fold-'
FILE_SUFFIX = '-ellipseList.txt'
IMAGE_PATH = '../../originalPics/'
TRAIN_POS_PATH = '../../savedPics/train/pos/'
TRAIN_NEG_PATH = '../../savedPics/train/neg/'
TEST_POS_PATH = '../../savedPics/test/pos/'
TEST_NEG_PATH = '../../savedPics/test/neg/'
countOfPos = 0
countOfNeg = 0
posSTR = 'pos'
negSTR = 'neg'
extensionSTR = ".jpg"
ellipseSTR = "ellipse"
resolution = 20
dice = 0
n = 1000
m = 100



# Crop positive and negative patches, then save
def cropAndSave(annotations, im):
	
	global countOfPos
	global countOfNeg
	global dice

	for ellipse in annotations:

		# Creating Rectangles from ellipse (inspiration from @hualitlc)
		ellipse = ellipse.split(' ')
		a = float(ellipse[0])
		b = float(ellipse[1])
		angle = float(ellipse[2])
		centre_x = float(ellipse[3])
		centre_y = float(ellipse[4])

		w = im.width
		h = im.height
		rectH = 2*a*(math.cos(math.radians(abs(angle))))
		rectW = 2*b*(math.cos(math.radians(abs(angle))))

		lx = float(max(0, centre_x - rectW/2))
		ly = float(max(0, centre_y - rectH/2))
		rx = float(min(w - 1, centre_x + rectW/2))
		ry = float(min(h - 1, centre_y + rectH/2))

		# Positive Patch
		bboxPos = (lx, ly, rx, ry)

		# Negative Patch
		left = random.randint(1, w//4)
		upper = random.randint(1, h//4)
		bboxNeg = (left, upper, left + resolution, upper + resolution)

		if(dice%5 != 0):
			# Save train sample

			# Positive Sample
			patch = im.crop(bboxPos)
			patch = patch.resize((resolution, resolution))
			patch.save(TRAIN_POS_PATH + posSTR + str(countOfPos) + extensionSTR, dpi = (resolution, resolution))
			countOfPos += 1

			# Negative Sample
			patch = im.crop(bboxNeg)
			patch = patch.resize((resolution, resolution))
			patch.save(TRAIN_NEG_PATH + negSTR + str(countOfNeg) + extensionSTR, dpi = (resolution, resolution))
			countOfNeg += 1

			dice += 1

		else:
			# Save test sample

			# Positive Sample
			patch = im.crop(bboxPos)
			patch = patch.resize((resolution, resolution))
			patch.save(TEST_POS_PATH + posSTR + str(countOfPos) + extensionSTR, dpi = (resolution, resolution))
			countOfPos += 1

			# Negative Sample
			patch = im.crop(bboxNeg)
			patch = patch.resize((resolution, resolution))
			patch.save(TEST_NEG_PATH + negSTR + str(countOfNeg) + extensionSTR, dpi = (resolution, resolution))
			countOfNeg += 1

			dice += 1





# Listing all files and processing one by one
files = []
for _, _, fs in os.walk(ANNOTATION_PATH):
	for filename in fs:
		if(ellipseSTR in filename):
			files.append(filename)
files.sort()

for k in tqdm(range(len(files)), unit = 'files'):
	file = files[k]
	try:
		with open(ANNOTATION_PATH + file) as f:
			lines = [line.rstrip('\n') for line in f]
	except:
		print('File not found!')
		break

	i = 0
	while(i < len(lines)):
		pathToImage = lines[i]
		im = Image.open(IMAGE_PATH + pathToImage + extensionSTR)
		i += 1
		numberOfFaces = int(lines[i])
		j = i + 1
		annotations = []
		while(j < i + 1 + numberOfFaces):
			annotations.append(lines[j])
			j += 1
		i = j
		
		cropAndSave(annotations, im)
