import cv2                 # working with, mainly resizing, images
import numpy as np         # dealing with arrays
import os                  # dealing with directories
from random import shuffle # mixing up or currently ordered data that might lead our network astray in training.
from tqdm import tqdm      # a nice pretty percentage bar for tasks. Thanks to viewer Daniel BA1/4hler for this suggestion
import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression


IMG_SIZE = 100
TRAIN_DIR = 'Newdataset'
LR = 1e-3

MODEL_NAME = 'facedetetion-{}-{}.model'.format(LR, '6conv-basic') # just so we remember which saved model is which, sizes must match

def label_img(label_name):
	word_label =[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
	word_label[int(label_name)-1]=1
	return word_label
	
def create_train_data():
	training_data = []
	for label_name in tqdm(os.listdir(TRAIN_DIR)):
		label = label_img(label_name)
		new_path = os.path.join(TRAIN_DIR,label_name)
		for img in tqdm(os.listdir(new_path)):
			path = os.path.join(new_path,img)
			img = cv2.imread(path,cv2.IMREAD_GRAYSCALE)
			img = cv2.resize(img, (IMG_SIZE,IMG_SIZE))
			cv2.imshow("Processing Img",img)
			cv2.waitKey(2)
			training_data.append([np.array(img),np.array(label)])
	shuffle(training_data)
	np.save('Newtrain_data.npy', training_data)
	return training_data

def make_model(train_x,train_y,test_x,test_y):
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
	model.fit({'input': X}, {'targets': Y}, n_epoch=7, validation_set=({'input': test_x}, {'targets': test_y}), 
		snapshot_step=1000, show_metric=True, run_id=MODEL_NAME)
	model.save('model/'+MODEL_NAME)
	print("Model saved completely")


#train_data=create_train_data()	
# If you have already created the dataset:
train_data = np.load('Newtrain_data.npy')

	
train = train_data[:-350]
test = train_data[-350:]

X = np.array([i[0] for i in train]).reshape(-1,IMG_SIZE,IMG_SIZE,1)
Y = [i[1] for i in train]

test_x = np.array([i[0] for i in test]).reshape(-1,IMG_SIZE,IMG_SIZE,1)
test_y = [i[1] for i in test]

make_model(X,Y,test_x,test_y)	