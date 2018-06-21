import time
import numpy as np
import tensorflow as tf
from keras.utils import to_categorical
import os
from model import Wavenet
from keras import optimizers, metrics
from keras import backend as K
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, CSVLogger, TensorBoard

X_train = {}
X_test = {}
Y_train = {}
Y_test = {}

path_Xtrain = "../../word_pos_large/X_train/"
path_Xtest = "../../word_pos_large/X_test/"
path_Ytrain = "../../word_pos_large/Y_train/"
path_Ytest = "../../word_pos_large/Y_test/"

all_words = os.listdir(path_Xtrain)

be_ = 'be.txt'
have_ = 'have.txt'
say_ = 'say.txt'
group_ = 'group.txt'
make_ = 'make.txt'

# for word in all_words:
#     X_train[word] = np.loadtxt(path_Xtrain+word, delimiter = ",")
	# Y_train[word] = np.loadtxt(path_Ytrain+word, delimiter = ",")
	# X_test[word] = np.loadtxt(path_Xtest+word, delimiter = ",")
	# Y_test[word] = np.loadtxt(path_Ytest+word, delimiter = ",")
word = 'have.txt'
X_train= np.loadtxt(path_Xtrain+word, delimiter = ",")
Y_train = np.loadtxt(path_Ytrain+word, delimiter = ",")
X_test = np.loadtxt(path_Xtest+word, delimiter = ",")
Y_test = np.loadtxt(path_Ytest+word, delimiter = ",")

# print("shape",X_train.shape)
X_train= X_train.reshape([X_train.shape[0],200,4])
X_train= np.transpose(X_train,[0,2,1])
X_test = X_test.reshape([X_test.shape[0],200,4])
X_test= np.transpose(X_test,[0,2,1])
# print(X_train.shape)
# X_train= np.transpose(X_train,[0,2,1])

# print (X_train.shape)

output_classes = len(set(Y_train))
print(set(Y_train))
print("classes",output_classes)
print (Y_train.shape)
# Y_train = np.array([to_categorical(i, num_classes=output_classes) for i in Y_train])
# Y_test = np.array([to_categorical(i, num_classes=output_classes) for i in Y_test])
Y_train = Y_train-1
Y_test = Y_test -1
Y_train = to_categorical(Y_train, num_classes=output_classes)
Y_test = to_categorical(Y_test, num_classes=output_classes)


opt_methods = {
	'sgd': optimizers.SGD(lr=0.01, momentum=0.0, decay=0.0, nesterov=True),
	'adadelta': optimizers.Adadelta(lr=1.0, rho=0.95, epsilon=1e-06),
	'adagrad':optimizers.Adagrad(lr=0.01, epsilon=1e-06),
	'adam':optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08),
	'adamax':optimizers.Adamax(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08),
	'nadam':optimizers.Nadam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, schedule_decay=0.004),
	'rmsprop': optimizers.RMSprop(lr=0.003, rho=0.9, epsilon=1e-08, decay=0.0)
}

training_params = {
	"steps_per_epoch": 100, 
	"number_year_to_train": 10, # number of years to train
	"end_year": 2000 # end year is usually 1985 + number_of_files + num_years_to_validate you wanna validate
}

model_params = {
	"range": 4,
	"input_depth": 200,
	"layers": 2, # deep network setup
	"num_filter": 8,
	'dilation_depth': 2,
	'output_classes': output_classes
}

def prepare_callbacks(model_file_path, test_folder, test_name, use_adaptive_optimzer=True):
	reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.2, patience=10, min_lr=0.000001, verbose=1, mode="min")
	
	model_name = "weights_{epoch:05d}.hdf5"
	saved_model_path = model_file_path
	save_chkpt = ModelCheckpoint(
					saved_model_path+test_folder+model_name,\
					verbose=1,\
					save_best_only=False,\
					monitor='loss',\
					mode='auto',\
					period=5
				)
	training_log_path = './'+"training_logs_"+test_name
	logger = CSVLogger(training_log_path)

	if use_adaptive_optimzer:
		callback_list = [logger, save_chkpt]
	else:   
		callback_list = [reduce_lr, logger, save_chkpt]
	
	return callback_list



def run_training(x_data, y_data, x_test, y_test, model_params, model_file_path, test_folder, test_name): 
	# set up the model
	
	def f1(y_true, y_pred):
		def recall(y_true, y_pred):
			"""Recall metric.

			Only computes a batch-wise average of recall.

			Computes the recall, a metric for multi-label classification of
			how many relevant items are selected.
			"""
			true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
			possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
			recall = true_positives / (possible_positives + K.epsilon())
			return recall

		def precision(y_true, y_pred):
			"""Precision metric.

			Only computes a batch-wise average of precision.

			Computes the precision, a metric for multi-label classification of
			how many selected items are relevant.
			"""
			true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
			predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
			precision = true_positives / (predicted_positives + K.epsilon())
			return precision

		precision = precision(y_true, y_pred)
		recall = recall(y_true, y_pred)
		return 2*((precision*recall)/(precision+recall+K.epsilon()))

	wavenet = Wavenet(**model_params)
	model = wavenet.model
	model.compile(loss='categorical_crossentropy', 
		optimizer=opt_methods['adam'], 
		metrics=[metrics.categorical_accuracy, f1])
	callback_list = prepare_callbacks(model_file_path, test_folder, test_name)

	print ("Start Training........")
	start_time = time.time()
	model.fit(x=x_data,
		y=y_data,
		batch_size=50,
		epochs = 10, 
		verbose = 1,
		validation_data=(x_test,y_test))
	duration = time.time() - start_time
	print ("Training duration: ", duration)

if __name__ == '__main__':
	model_file_path = './saved_model/'
	evaluation_file_path = './evaluation_result/'
	test_name = word
	test_folder = 'No_dividend_adj/'+test_name+'/'

	# print (model_params)
	run_training(X_train, Y_train, X_test, Y_test, model_params, model_file_path, test_folder, test_name)