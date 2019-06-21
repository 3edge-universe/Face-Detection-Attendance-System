import numpy as np
import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
import NameFind
import cv2
import math

IMG_SIZE = 100
LR = 1e-3

MODEL_NAME = 'facedetetion-{}-{}.model'.format(LR, '6conv-basic') # just so we remember which saved model is which, sizes must match
face_cascade = cv2.CascadeClassifier('Haar/haarcascade_frontalcatface.xml')
eye_cascade = cv2.CascadeClassifier('Haar/haarcascade_eye.xml')
glass_cas = cv2.CascadeClassifier('Haar/haarcascade_eye_tree_eyeglasses.xml')


def load_model(train_x,train_y,test_x,test_y):
	convnet = input_data(shape=[None, IMG_SIZE, IMG_SIZE, 1], name='input')

	convnet = conv_2d(convnet, 32, 5, activation='relu')
	convnet = max_pool_2d(convnet, 5)

	convnet = conv_2d(convnet, 64, 5, activation='relu')
	convnet = max_pool_2d(convnet, 5)


	# 3 extra cov layers
	convnet = conv_2d(convnet, 128, 5, activation='relu')
	convnet = max_pool_2d(convnet, 5)

	convnet = conv_2d(convnet, 64, 5, activation='relu')
	convnet = max_pool_2d(convnet, 5)

	convnet = conv_2d(convnet, 32, 5, activation='relu')
	convnet = max_pool_2d(convnet, 5)
	#here extra layers are removed


	convnet = fully_connected(convnet, 1024, activation='relu')
	convnet = dropout(convnet, 0.8)

	convnet = fully_connected(convnet, 20, activation='softmax')
	convnet = regression(convnet, optimizer='adam', learning_rate=LR, loss='categorical_crossentropy', name='targets')

	model = tflearn.DNN(convnet, tensorboard_dir='log')
	model.load('model/'+MODEL_NAME)
	return model

#testing on run time data

def DetectEyes(Image):
	Theta = 0
	rows, cols = Image.shape
	glass = glass_cas.detectMultiScale(Image)                                               # This ditects the eyes
	for (sx, sy, sw, sh) in glass:
		if glass.shape[0] == 2:                                                             # The Image should have 2 eyes
			if glass[1][0] > glass[0][0]:
				DY = ((glass[1][1] + glass[1][3] / 2) - (glass[0][1] + glass[0][3] / 2))    # Height diffrence between the glass
				DX = ((glass[1][0] + glass[1][2] / 2) - glass[0][0] + (glass[0][2] / 2))    # Width diffrance between the glass
			else:
				DY = (-(glass[1][1] + glass[1][3] / 2) + (glass[0][1] + glass[0][3] / 2))   # Height diffrence between the glass
				DX = (-(glass[1][0] + glass[1][2] / 2) + glass[0][0] + (glass[0][2] / 2))   # Width diffrance between the glass

			if (DX != 0.0) and (DY != 0.0):                                                 # Make sure the the change happens only if there is an angle
				Theta = math.degrees(math.atan(round(float(DY) / float(DX), 2)))            # Find the Angle
				
				M = cv2.getRotationMatrix2D((cols / 2, rows / 2), Theta, 1)                 # Find the Rotation Matrix
				Image = cv2.warpAffine(Image, M, (cols, rows))
				#cv2.imshow('ROTATED', Image)                                              # UNCOMMENT IF YOU WANT TO SEE THE

				Face2 = face_cascade.detectMultiScale(Image, 1.3, 5)                                # This detects a face in the image
				for (FaceX, FaceY, FaceWidth, FaceHeight) in Face2:
					CroppedFace = Image[FaceY: FaceY + FaceHeight, FaceX: FaceX + FaceWidth]
					return CroppedFace

train_data = np.load('Newtrain_data.npy')
train = train_data[:-350]
test = train_data[-350:]

X = np.array([i[0] for i in train]).reshape(-1,IMG_SIZE,IMG_SIZE,1)
Y = [i[1] for i in train]

test_x = np.array([i[0] for i in test]).reshape(-1,IMG_SIZE,IMG_SIZE,1)
test_y = [i[1] for i in test]

model=load_model(X,Y,test_x,test_y)

def find_max(array):
	lt=[]
	for i in range(3):
		max=array.argmax()
		lt.append(max)
		array[max]=0
	return lt
				
def predict():					
	cap = cv2.VideoCapture(0)	# Camera object
	c=1
	while c:
		ret, img = cap.read()
		gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)                                                    # Convert the Camera to graySe
		if np.average(gray) > 110:                                                                      # Testing the brightness of the image
			faces = face_cascade.detectMultiScale(gray, 1.3, 5)                                         # Detect the faces and store the positions
			for (x, y, w, h) in faces:                                                                  # Frames  LOCATION X, Y  WIDTH, HEIGHT
				FaceImage = gray[y - int(h / 2): y + int(h * 1.5), x - int(x / 2): x + int(w * 1.5)]    # The Face is isolated and cropped
				Img = (DetectEyes(FaceImage))
				IMG=cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),3)
				if Img is not None:
					frame = Img                                                                         # Show the detected faces
				else:
					frame = gray[y: y+h, x: x+w]
				frame = cv2.resize(frame, (IMG_SIZE,IMG_SIZE))
				cv2.imshow("CAPTURED PHOTO", IMG)                                                 # show the captured image
				cv2.waitKey(1)
				data = frame.reshape(IMG_SIZE,IMG_SIZE,1)
				output=model.predict([data])[0]
				max_index=output.argmax()
				if output[max_index-1]>0.5:
					lt=find_max(output)
					print('Choose Who are you\n1.',NameFind.take_name(lt[0]+1))
					print('2.',NameFind.take_name(lt[1]+1))
					print('3.',NameFind.take_name(lt[2]+1))
					print('4.None of the above and 5.Exit')
					choice=int(input('Write Your choice?	'))
					if choice ==5:
						exit(0)
					elif choice is not 4:
						Name=NameFind.take_attendance(lt[choice-1]+1)
						c=0
						print("Your Attendance in Taken --",Name)
				else:
					print("You Are Unknown to System")
					c=0
	cap.release()
	cv2.destroyAllWindows()
	
predict()