import cv2
import numpy as np
import csv
import math
import sys
import os
import itertools
from decimal import Decimal, ROUND_HALF_UP

#grader.py ANSWER_SHEET KEY STUDENT_NAME

#pixel definitions for 300 dpi
x_min = 300
x_max = 2230
y_min = 850
y_max = 2460
area_min = 800
area_max = 1400

######################################## ALIGN IMAGE ###############################################

#read images
blank_sheet = cv2.imread('blankSheet.jpg', cv2.IMREAD_GRAYSCALE)
bubbled_sheet = cv2.imread(sys.argv[1], cv2.IMREAD_GRAYSCALE)


def detect_squares(im):
    #set parameters for blob detector
    params = cv2.SimpleBlobDetector_Params()

    params.filterByArea = True;
    params.minArea = 1150;
    params.filterByCircularity = True;
    params.maxCircularity = .9;
    params.minCircularity = .55;

    #set up the detector with parameters
    detector = cv2.SimpleBlobDetector_create(params)

    #detect squares
    keypoints = detector.detect(im)
    refPoints = [0,1,2]

    for final_key in keypoints:
        if final_key.pt[0] < 300:
            if final_key.pt[1] < 300:
                refPoints[0] = [final_key.pt[0], final_key.pt[1]]
            if final_key.pt[1] > 2000:
                refPoints[1] = [final_key.pt[0], final_key.pt[1]]
        if final_key.pt[0] > 2000:
            if final_key.pt[1] > 2000:
                refPoints[2] = [final_key.pt[0], final_key.pt[1]]

    return refPoints

#call detect_squares function to find the three points for warpAffine
points_src = np.float32(detect_squares(bubbled_sheet))
points_dst = np.float32(detect_squares(blank_sheet))

#find size of image
image_size = blank_sheet.shape

#generate affine transformation matrix
warp_matrix = cv2.getAffineTransform(points_src, points_dst)

#use warpAffine to align images
bubbled_aligned = cv2.warpAffine(bubbled_sheet, warp_matrix, (image_size[1], image_size[0])) #, flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP


######################################## READ IN CORRECT KEY ########################################

dictNames = ["English","Math","Reading","Science"]

keyDictionary = {dictNames[0]:[], dictNames[1]:[], dictNames[2]:[], dictNames[3]:[]}
with open('keys/%s' % (sys.argv[2]), 'r') as csvFile:
    keyReader = csv.reader(csvFile, delimiter=',')
    counter = 0
    for row in keyReader:
        keyDictionary[dictNames[counter]].append(row)
        counter += 1

english_key = list(map(int, keyDictionary["English"][0]))
math_key = list(map(int, keyDictionary["Math"][0]))
reading_key = list(map(int, keyDictionary["Reading"][0]))
science_key = list(map(int, keyDictionary["Science"][0]))

final_bubbled = []
score = 0

######################################## READ IMAGE AND FIND BUBBLES ########################################

#read image, apply Gaussian blur, apply edge detection
blurred = cv2.GaussianBlur(blank_sheet, (5, 5), 0)
edged = cv2.Canny(blurred, 75, 200)

#apply Otsu's thresholding method to binarize image
thresh = cv2.threshold(blank_sheet, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

#detect contours
contours = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contours = contours[0]

#initialize array to hold bubbles
bubbles = []
properties = []
counter = 0
bounding_boxes = []

#iterate over contour points to bound them in rectangles and detect bubbles 
#of the correct size to add to bubbles array
for point in contours:
	(x, y, w, h) = cv2.boundingRect(point)
	aspectRatio = w/float(h)
	area = w*h
	#print(area)
	
	if (x > x_min and x < x_max and y > y_min and y < y_max and area > area_min and area < area_max):
		bubbles.append(point)
		bounding_boxes.append([x, y, w, h, point])


######################################## SORT BUBBLES AND MAKE QUESTION ARRAYS ########################################

#sort each of the bounding boxes by the y-coordinate
bounding_boxes = sorted(bounding_boxes, key=lambda box:box[1])
sorted_boxes = []
english_questions = []
math_questions = []
reading_questions = []
science_questions = []

#function to sort the bubbles
def questionSort(start, stop, step):
	for index in range(start,stop,step):
		boxes = sorted(bounding_boxes[index:index+step])
		for box in boxes:
			sorted_boxes.append(box)

#function to group data from array into blocks (questions)
def grouper(iterable, n, question_array, fillvalue=None):
	args = [iter(iterable)] * n
	question_list = itertools.zip_longest(*args, fillvalue=fillvalue)
	for question in question_list:
		question_array.append(question)

#sort the rows of the English section
questionSort(0,240,24)
questionSort(240,300,20)

#sort the rows of the math section
questionSort(300,600,30)

#sort the rows of the reading section
questionSort(600,720,24)
questionSort(720,760,20)

#sort the questions of the science section
questionSort(760,880,24)
questionSort(880,920,20)

#call grouper to group sets of questions into arrays,
#then store those into arrays for each section
grouper(sorted_boxes[0:300], 4, english_questions)
grouper(sorted_boxes[300:600], 5, math_questions)
grouper(sorted_boxes[600:760], 4, reading_questions)
grouper(sorted_boxes[760:920], 4, science_questions)

#sort each question array so the questions are in numerical order
english_questions = sorted(english_questions, key=lambda box:box[0][0])
math_questions = sorted(math_questions, key=lambda box:box[0][0])
reading_questions = sorted(reading_questions, key=lambda box:box[0][0])
science_questions = sorted(science_questions, key=lambda box:box[0][0])

english_bubbled = []
math_bubbled = []
reading_bubbled = []
science_bubbled = []

thresh = cv2.threshold(bubbled_aligned, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

######################################## FUNCTION TO GRADE TEST ########################################

def grader(section, section_key, section_bubbled):

	reading = {
		'Context': 0,
		'Detail': 0,
		'Interference': 0,
		'Main Idea': 0,
		'Structure': 0,
		'Tone': 0,
		'Vocab in Context': 0
	}

	english = {
		'Agreement': 0,
		'Clarity': 0,
		'Comprehension': 0,
		'Conjunctions': 0,
		'Idioms': 0,
		'Phrasing': 0,
		'Pronouns': 0,
		'Punctuation': 0,
		'Run-on/Fragment': 0,
		'Verbs': 0,
		'Word Choice': 0
	}

	science = {
		'Easy': 0,
		'Average': 0,
		'Difficult': 0
	}

	section_score = 0

	with open('pixels.csv', 'w') as csvfile: 
		for (question_number, question) in enumerate(section):
			csv_writer = csv.writer(csvfile, delimiter=',')

			bubbled = None
			pixel_count = []

			for (i, answers) in enumerate(question):

				mask = np.zeros(thresh.shape, dtype="uint8")
				cv2.drawContours(mask, [answers[4]], -1, 255, -1)
				mask = cv2.bitwise_and(thresh, thresh, mask=mask)
				total = cv2.countNonZero(mask)

				#csv_writer.writerow([question_number, total] )
	
				if bubbled is None or total > bubbled[0]:
					bubbled = (total, i)

				pixel_count.append(total)

			#check to see if a question was left blank
			#append the bubbled array with a 5 if so
			if (np.std(pixel_count) < 40): #and np.ptp(pixel_count) < 100):
				section_bubbled.append(5)
				bubbled = (100000, 100000)
			else:
				section_bubbled.append(bubbled[1])

			if (section_key[question_number] == bubbled[1]):
				section_score += 1
		
			csv_writer.writerow([question_number, pixel_count[0], pixel_count[1], pixel_count[2], pixel_count[3], np.std(pixel_count)])

	return section_score


######################################## GENERATE PERCENTILE RANKINGS ########################################

def percents(scoreArray):

	percentArray = []

	with open('percentiles.csv', 'r') as csvFile:
		scaleReader = csv.reader(csvFile, delimiter=',')
		for row in scaleReader:
			percentArray.append(row)

	finalPercents = [percentArray[0][scoreArray[0]-1], percentArray[1][scoreArray[1]-1], percentArray[2][scoreArray[2]-1], percentArray[3][scoreArray[3]-1], percentArray[4][scoreArray[4]-1]]

	return finalPercents


######################################## CALL GRADING FUNCTION AND OUTPUT RESULTS TO LATEX FILE ########################################

def alphanumeric(section_key, section_bubbled):

	ansChoicesO = ['A','B','C','D','E','?']
	ansChoicesE = ['F','G','H','J','K','?']

	for i in range(0, len(section_key)):
		if (i % 2) == 0:
			section_bubbled[i] = ansChoicesO[section_bubbled[i]]
			section_key[i] = ansChoicesO[section_key[i]]
		else: 
			section_bubbled[i] = ansChoicesE[section_bubbled[i]]
			section_key[i] = ansChoicesE[section_key[i]]

def scaler(numCorrect,index):

	scaleArray = []

	with open('keys/%s_scale.csv' % (sys.argv[2]), 'r') as csvFile:
		scaleReader = csv.reader(csvFile, delimiter=',')
		for row in scaleReader:
			scaleArray.append(row)

	if (index == 0):
		scaleDict = {i:scaleArray[0][i] for i in range(0,76)}
	if (index == 1):
		scaleDict = {i:scaleArray[1][i] for i in range(0,61)}
	if (index == 2):
		scaleDict = {i:scaleArray[2][i] for i in range(0,41)}
	if (index == 3):
		scaleDict = {i:scaleArray[3][i] for i in range(0,41)}

	return int(scaleDict[numCorrect])

with open('bubbles.csv', 'w') as file:
	english_score = grader(english_questions, english_key, english_bubbled)
	math_score = grader(math_questions, math_key, math_bubbled)
	reading_score = grader(reading_questions, reading_key, reading_bubbled)
	science_score = grader(science_questions, science_key, science_bubbled)

alphanumeric(english_key, english_bubbled)
alphanumeric(math_key, math_bubbled)
alphanumeric(reading_key, reading_bubbled)
alphanumeric(science_key, science_bubbled)

final_scores = [scaler(english_score, 0), scaler(math_score, 1), scaler(reading_score, 2), scaler(science_score, 3)]
final_scores.append(int(Decimal(sum(final_scores)/float(4)).to_integral_value(rounding=ROUND_HALF_UP)))
final_percents = percents(final_scores)
final_bubbled = [english_bubbled, math_bubbled, reading_bubbled, science_bubbled]
final_key = [english_key, math_key, reading_key, science_key]

section = ["English", "Math", "Reading", "Science"]
col_size = [13, 10, 7, 7]

#print(len(final_bubbled[1]))

with open('answers.tex','w') as scoreReport:

	scoreReport.write("\\documentclass[10pt]{article}\n")
	scoreReport.write("\\usepackage{amsmath,amsthm,verbatim,amssymb,amsfonts,amscd,graphicx,graphics,fullpage,color,colortbl}\n")
	scoreReport.write("\\definecolor{LightCyan}{rgb}{0.65,1,1}\n")
	scoreReport.write("\\newcolumntype{g}{>{\columncolor{LightCyan}}c}\n")
	scoreReport.write("\\begin{document}\n\n")
	scoreReport.write("\\noindent\\includegraphics[scale=.6]{logo.eps}")
	scoreReport.write("\\qquad\\hspace{1in}")
	scoreReport.write("\\begin{tabular}{|l|c|c|}\\hline")
	scoreReport.write("\multicolumn{3}{|l|}{%s}\\\ \n \\hline \n" % (sys.argv[3]))
	scoreReport.write("Section & Score & Percentile\\\ \n \\hline \n")
	for x in range(0,4):
		scoreReport.write("%s & %d & %s \\\ \n" % (section[x],final_scores[x],final_percents[x]))
	scoreReport.write("\\hline \n Composite & %d & %s \\\ \n \\hline \n" % (final_scores[4],final_percents[4]))
	scoreReport.write("\\end{tabular}\n")

	for x in range(0,4):
		scoreReport.write("\\section{%s} \n" % (section[x]))
		scoreReport.write("\\footnotesize")
		scoreReport.write("\\begin{tabular}{c|c|c} \n")
		scoreReport.write("$\\#$ & S & C \\\ \n \\hline \n ")
		
		for n in range(0, len(final_bubbled[x])):
			if final_bubbled[x][n] is final_key[x][n]:
				scoreReport.write("%d & %s & \\checkmark \\\ \n" % (n+1, final_bubbled[x][n]))
			else:
				scoreReport.write(" \\cellcolor{LightCyan} %d & %s & %s \\\ \n" % (n+1, final_bubbled[x][n], final_key[x][n]))
			if ((n+1) % col_size[x] == 0 and not (x == 1 and n == 59)):
				scoreReport.write("\\end{tabular} \n \\quad \n")
				scoreReport.write("\\begin{tabular}{c|c|c} \n")
				scoreReport.write("$\\#$ & S & C \\\ \n \\hline \n ")
		if x == 0:
			scoreReport.write("\\multicolumn{3}{c}{} \\\ \n \\multicolumn{3}{c}{} \\\ \n \\multicolumn{3}{c}{} \n")
		if x == 2 or x == 3:
			scoreReport.write("\\multicolumn{3}{c}{} \\\ \n \\multicolumn{3}{c}{} \n")

		scoreReport.write("\\end{tabular} \n")

	scoreReport.write("\\end{document}")

os.system("pdflatex -interaction=batchmode answers.tex")

grader(english_questions, english_key, english_bubbled)






#cv2.drawContours(image, bubbles, -1, (0,0,255), 6)
#v2.namedWindow("Keypoints", cv2.WINDOW_NORMAL)
#cv2.resizeWindow("Keypoints", 1000,1000)
#cv2.imshow("Keypoints", image)
#cv2.waitKey(0)
