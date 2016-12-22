from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D
from keras.models import model_from_json
import matplotlib.pyplot as plt
import numpy 
import h5py
import os
import theano
import matplotlib
from keras.models import load_model

# fix random seed for reproducibility
seed = 8
numpy.random.seed(seed)


#load data
(X_train,y_train), (X_test, y_test) = mnist.load_data()

# Reshape data  to [sample][channels][width][height]
#1 because MNIST data image are gray in color
#28 * 28 because its dimension of image

X_train = X_train.reshape(X_train.shape[0],1,28,28).astype('float32')
X_test = X_test.reshape(X_test.shape[0],1,28,28).astype('float32')


#normalize inputs from 0-255 to 0-1
X_train = X_train/255
X_test = X_test/255

#since its a multiclass classification problem so its good to use a hot encoding of the classs value. ie transforming the vecotr of class
#integer into a binary matrix

y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
num_classes = y_test.shape[1]

numberoffilters = 32; 

#define a CNN model
def baseline_model():
	model = Sequential()
	model.add(Convolution2D(numberoffilters,5,5, border_mode ='valid',subsample=(1,1),input_shape=(1,28,28), activation='relu'))

	model.add(MaxPooling2D(pool_size=(2,2)))
	
	model.add(Convolution2D(32,3,3, activation='relu'))
	
	model.add(MaxPooling2D(pool_size=(2,2)))


	model.add(Dropout(0.2))
	model.add(Flatten())
	model.add(Dense(128, activation='relu'))
	model.add(Dense(50, activation='relu'))
	model.add(Dense(num_classes, activation='softmax'))
	#compile model
	model.compile(loss='categorical_crossentropy',optimizer='adam', metrics=['accuracy'])
	return model


if os.path.isfile('/home/naruto/Documents/crashcourse/model.h5'):
	# load json and create model
	json_file = open( 'model.json' , 'r' )
	loaded_model_json = json_file.read()
	json_file.close()
	loaded_model = model_from_json(loaded_model_json)
	# load weights into new model
	loaded_model.load_weights("model.h5")
	print("Loaded model from disk")
	# evaluate loaded model on test data
	#loaded_model.compile(loss= 'binary_crossentropy' , optimizer= 'rmsprop' , metrics=['accuracy'])
	loaded_model.compile(loss='categorical_crossentropy',optimizer='adam', metrics=['accuracy'])
	score = loaded_model.evaluate(X_test, y_test, verbose=0)
	print "%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100)
	model = loaded_model
else:
	#build the model 
	model = baseline_model()
	#Fit model
	model.fit(X_train,y_train, validation_data =(X_test,y_test), nb_epoch=10, batch_size=200, verbose=2)
	#model.save('mymnistdatavisulization.h5')

	#model = load_model('/home/naruto/Documents/crashcourse/mymnistdatavisualization.h5')

	#Final evaluation of the model
	scores = model.evaluate(X_test,y_test,verbose=0)
	model.summary()
	print("Baseline Error: %.2f%%"%(100-scores[1]*100))

	#serialize model to JSON
	model_json = model.to_json()
	with open("model.json", "w") as json_file:
		json_file.write(model_json)
	# serialize weights to HDF5
	model.save_weights("model.h5")
	print("Saved model to disk")


#print(model.layers[0].get_config())

#print(model.layers)

#print(model.inputs)
print(model.outputs)
##viewing the output of each filters
def plot_filters(layer, x , y):
	#of which layer filter you want to visualize ---> layers
	#x----> number of image in rows
	#y---> number of image in column
	filters = layer.W.get_value()
	fig = plt.figure()
	for j in range(len(filters)):
		ax = fig.add_subplot(y, x, j+1)
		ax.matshow(filters[j][0], cmap = matplotlib.cm.binary)
		plt.xticks(numpy.array([]))
		plt.yticks(numpy.array([]))
	plt.tight_layout()
	return plt

plot = plot_filters(model.layers[0],8,8)
plot.show()


target_layer = 1


for i in range(4):
	#visualizing the intermediate layers
	output_layer = model.layers[i].output
	output_fn = theano.function([model.layers[0].input],output_layer)

	print(i);

	#input image
	input_image = X_train[4:5, :, :, :]
	plt.imshow = input_image
	print(input_image.shape)
	plt.show()

	#feeding output image
	output_image = output_fn(input_image)
	print(output_image.shape)


	#rearrnging dimension so we can plot the result as RGB image
	output_image  = numpy.rollaxis(numpy.rollaxis(output_image, 3 , 1), 3, 1)
	print(output_image.shape)

	fig = plt.figure(figsize=(8,8))

	for i in range(numberoffilters):
		ax = fig.add_subplot(8, 8, i+1)
		ax.imshow(output_image[0,:,:,i],cmap = matplotlib.cm.gray)
		plt.xticks(numpy.array([]))
		plt.yticks(numpy.array([]))
		plt.tight_layout()
	plt.show()
model.summary()


