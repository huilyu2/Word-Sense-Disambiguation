# This file serves the purpose of train models.

import numpy as np
import tensorflow as tf
from keras.utils import to_categorical
import os
from model import wavenet_model

X_train = {}
X_test = {}
Y_train = {}
Y_test = {}

path_Xtrain = "../../upsampled/X_train/"
path_Xtest = "../../upsampled/X_test/"
path_Ytrain = "../../upsampled/Y_train/"
path_Ytest = "../../upsampled/Y_test/"

all_words = os.listdir(path_Xtrain)

be_ = 'be.txt'
have_ = 'have.txt'
say_ = 'say_.txt'
group_ = 'group.txt'
make_ = 'make.txt'

# for word in all_words:
#     X_train[word] = np.loadtxt(path_Xtrain+word, delimiter = ",")
    # Y_train[word] = np.loadtxt(path_Ytrain+word, delimiter = ",")
    # X_test[word] = np.loadtxt(path_Xtest+word, delimiter = ",")
    # Y_test[word] = np.loadtxt(path_Ytest+word, delimiter = ",")
word = group_
X_train= np.loadtxt(path_Xtrain+word, delimiter = ",")
Y_train = np.loadtxt(path_Ytrain+word, delimiter = ",")
X_test = np.loadtxt(path_Xtest+word, delimiter = ",")
Y_test = np.loadtxt(path_Ytest+word, delimiter = ",")

# print("shape",X_train.shape)
X_train= X_train.reshape([X_train.shape[0],100,4])
# print(X_train.shape)
X_train= np.transpose(X_train,[0,2,1])

# print (X_train.shape)

output_classes = len(set(Y_train))
Y_train = to_categorical(Y_train, num_classes=output_classes)

# print(Y_train.shape)

# # if __name__ == '__main__':
# #     print (X_train.shape)


# training params
initialize = True
iteration_start = 0
steps = 12000
batch_size = 1
data_size = X_train.shape[0]

# model params
params = {
	'batch_size':1,
    'dilations':[1,2,1,2],
    'filter_width':2,
    'residual_filters':8,
    'dilation_filters':8,
    'skip_filters':32,
    'input_channels':100,
    'output_classes': output_classes,
    'use_biases': False,
    'global_condition_channels': None,
    'global_condition_cardinality': None
}


net = wavenet_model(
    params['batch_size'],
    params['dilations'],
    params['filter_width'],
    params['residual_filters'],
    params['dilation_filters'],
    params['skip_filters'],
    params['input_channels'],
    params['output_classes'],
    params['use_biases'],
    params['global_condition_channels'],
    params['global_condition_cardinality'])

x=tf.placeholder(tf.float32,shape=(None,4,100))
y=tf.placeholder(tf.float32,shape=(None,output_classes))
loss = net.loss(input_batch=x,
                target_output = y,
                global_condition_batch=None,
                l2_regularization_strength=None)
train_step=tf.train.AdamOptimizer(0.001).minimize(loss)

tf.summary.scalar("loss", loss)
summary_op = tf.summary.merge_all()
# trainable = tf.trainable_variables()

initializer = tf.global_variables_initializer()
saver = tf.train.Saver(max_to_keep = 10)

with tf.Session() as sess:

    if initialize:
        sess.run(initializer)
    else:
        ckpt_file = './models/params_' + str(iteration_start) + '.ckpt'
        print('restoring parameters from', ckpt_file)
        print("Model restored.")
        saver.restore(sess, ckpt_file)

    # create log writer object
    logs_path = './train_logs/logs'
    train_writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())
    # train_writer = tf.summary.FileWriter(logs_path, graph=sess.graph)
    # train_writer.add_graph(sess.graph)


    for i in range(iteration_start, steps): ##################

        #选定每一个批量读取的首尾位置，确保在1个epoch内采样训练
        start = i * batch_size % data_size
        end = min(start + batch_size,data_size)
        _, summary = sess.run([train_step,summary_op],feed_dict={x:X_train[start:end],y:Y_train[start:end]})
        # writer.add_summary(summary, epoch * batch_count + i)
        train_writer.add_summary(summary, i)
        if i % 1000 == 0:
            training_loss= sess.run(cross_entropy,feed_dict={x:X,y:Y})
            print("By %d iteration，Training loss is %g"%(i,training_loss))
            # save the model
            saver.save(sess, './models/params_' + str(i) + '.ckpt')