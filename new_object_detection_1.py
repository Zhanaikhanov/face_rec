import cv2
# pip install opencv-contrib-python
import numpy as np
import random
import os
face_cascade = cv2.CascadeClassifier('/home/djangoman/erzhan/lib/python3.5/site-packages/cv2/data/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('/home/djangoman/erzhan/lib/python3.5/site-packages/cv2/data/haarcascade_eye.xml')
# cap = cv2.VideoCapture(0)

print("0 - Web camera")
ere = input("or type video filename here as (input.avi):")
cap = None
if ere == "0":
	cap = cv2.VideoCapture(0)
else:
	cap = cv2.VideoCapture(ere)
# 

asd = open("persons.txt",'r')
persons = [line.strip() for line in asd]
asd.close()

print('stage 1...')

def face_detect_from(img):

	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	faces = face_cascade.detectMultiScale(gray, 1.3, 5)
	if len(faces) == 0:
		return None, None
	(x, y, w, h) = faces[0]
	return gray[y:y+h,x:x+w], faces[0]

def prediction(img):
	img = img.copy()
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	faces1 = face_cascade.detectMultiScale(gray, 1.3, 5)
	persons_name = []
	for face in faces1:
 
		(x, y, w, h) = face
		gr1 = gray[y:y+h,x:x+w]
		labels, conf = face_recognizer.predict(gr1)
		print("--->",conf)
		if conf>80:
			labels = 0
		person_name = persons[labels]
		cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 1)
		cv2.putText(img, person_name, (face[0], face[1]-5), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)
		persons_name.append(person_name)


	return img, faces1, persons_name

def prepare_training_data():
	dirs = os.listdir("training_data")
	faces = []
	labels = []
	for dir_name in dirs:
		if not dir_name.startswith("p"):
			continue
		label = int(dir_name.replace("p",""))
	
		person_dir_path = "training_data"+"/"+dir_name
		person_images_names = os.listdir(person_dir_path)

		for image_name in person_images_names:
			if image_name.startswith("."):
				continue
			image_path = person_dir_path+"/"+image_name

			image = cv2.imread(image_path)

			face, rect = face_detect_from(image)
			if face is not None:
				faces.append(face)
				labels.append(label)


	return faces, labels


# faces, labels = None
# face_recognizer = None
# face_recognizer.train(faces, np.array(labels))

feature_params = dict( maxCorners = 100,qualityLevel = 0.3,minDistance = 7,blockSize = 7 )
lk_params = dict( winSize  = (15,15),maxLevel = 2,criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
color = np.random.randint(0,255,(100,3))


recta = None
pers_name = None
errors = 0
ff = None

# ----------
wrt = open("persons.txt",'w')
try:
		
		
	while True:
		print(persons)
		print('====================================================')
		print('2-exit')	
		print('1-test')
		print('0-train')
		mode = int(input('Type command-->'))

		if mode == 1:
			faces, labels = prepare_training_data()
			face_recognizer = cv2.face.LBPHFaceRecognizer_create()
			face_recognizer.train(faces, np.array(labels))


		if mode == 2:
			break

		pp_name = ""
		path = 'training_data/p'

		if mode == 0:
			pp_name = input('Person\'s name-->')

			if pp_name in persons:
				path += str(persons.index(pp_name))
			else:
				persons.append(pp_name)
				path += str(len(persons)-1)


			if not os.path.exists(path):
			    os.makedirs(path)


		# ----------
		# 1 - test
		# 0 - train

		while mode == 0:
			ret, frame = cap.read()
			if not frame is None:
				if not ret:
					continue

				frame = cv2.flip(frame,1)

				cv2.imwrite(os.path.join(path , str(random.choice(range(100000,999999)))+'.jpg'), frame)
				cv2.imshow('mirror',frame)


			k = cv2.waitKey(30) & 0xFF
			if k == 27:
				break

		while (mode == 1):
			ret, frame = cap.read()

			if not frame is None:
				if not ret:
					continue
				frame = cv2.flip(frame,1)

				ff,ff1,ff2 = prediction(frame)
				# img, faces1, persons_name

				cv2.imshow('mirror',ff)
				k = cv2.waitKey(30) & 0xFF
				if k == 27:
					break


		cv2.destroyAllWindows()

	# out.release()
	cap.release()
	cv2.destroyAllWindows()

	wrt.write('\n'.join(persons))
	wrt.close()
except:
	cap.release()
	cv2.destroyAllWindows()

	wrt.write('\n'.join(persons))
	wrt.close()
