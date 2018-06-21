import keras
from keras import optimizers
from keras.initializers import TruncatedNormal
from keras.layers import Dense, Activation, merge, Input, BatchNormalization
from keras.layers.convolutional import Conv1D, Conv2D
from keras.models import Sequential, Model
from keras.models import load_model
from keras.constraints import maxnorm
from keras.layers.core import Activation, Flatten, Dense, Lambda
from keras.layers.advanced_activations import LeakyReLU, PReLU
from keras.layers import add, multiply, pooling

class Wavenet(object):

	def __init__(self, **params):
		"""
		Wavenet model.

			Args:
				input_shape (tuple of integers):
					input dimension

				nb_filters (int):
					how many filters in each layer
				
				output_classes (int):
					the depth of output

				dilation_depth (int):
				
				nb_stacks (int):
				
				use_skip_connections (bool):
				
				learn_all_outputs (bool):
				
				use_bias (bool):
		"""

		model_params = {
			'input_shape': (params['range'],params['input_depth']),
			'nb_filters': params['num_filter'],
			'output_classes': params['output_classes'],
			'dilation_depth': params['dilation_depth'],
			'nb_stacks': params['layers'],
			'use_skip_connections': True,
			'learn_all_outputs': True,
			'use_bias': True
		}

		x, y = self.build_wavenet(**model_params)
		
		self.model = Model(inputs=x, outputs=y)

	def build_wavenet(self,
					  input_shape,
					  nb_filters,
					  output_classes,
					  dilation_depth,
					  nb_stacks, 
					  use_skip_connections=True,
					  learn_all_outputs=True,
					  use_bias=True):
		# dilation_depth is how many layers inside a single block
		# nb_stacks is how many blocks in this model

		def residual_block(x):
			original_x = x
			tanh_out = Conv1D(nb_filters, 
							  2, 
							  dilation_rate=2 ** i,
							  # dilation_rate=2 ** 1,
							  padding='causal',
							  use_bias=use_bias, 
							  # name='dilated_conv_%d_tanh_s%d' % (2 ** 1, s),
							  # name='dilated_conv_%d_tanh_s%d' % (2 ** 1, nb_stacks),
							  activation='tanh'
							  # kernel_constraint=maxnorm(2.)
							  )(x)
			sigm_out = Conv1D(nb_filters,
							  2,
							  dilation_rate=2 ** i,
							  # dilation_rate=2 ** 1, 
							  padding='causal',
							  use_bias=use_bias,
							  # name='dilated_conv_%d_sigm_s%d' % (2 ** 1, s),
							  # name='dilated_conv_%d_tanh_s%d' % (2 ** 1, nb_stacks),
							  activation='sigmoid'
							  # kernel_constraint=maxnorm(2.)
							  )(x)
			x = multiply([tanh_out, sigm_out])

			res_x = Conv1D(nb_filters, 1, padding='same', use_bias=use_bias,kernel_constraint=maxnorm(2.))(x)
			skip_x = Conv1D(nb_filters, 1, padding='same', use_bias=use_bias,kernel_constraint=maxnorm(2.))(x)
			res_x = add([original_x, res_x])
			return res_x, skip_x
		
		print("In wavenet, input shape is ",input_shape)
		input = Input(shape=input_shape, name='input_layer')
		out = input
		skip_connections = []
		out = Conv1D(nb_filters, 2, dilation_rate=1, padding='causal',name='initial_causal_conv')(out)
		
		print("build number of blocks/stacks: ", nb_stacks)
		print("build number of dilation layers within one block/stack: ", dilation_depth)

		for s in range(nb_stacks): # how many blocks
			for i in range(0, dilation_depth + 1):
				out, skip_out = residual_block(out)
				skip_connections.append(skip_out)
		
		out, skip_out = residual_block(out)
		skip_connections.append(skip_out)

		if use_skip_connections:
			out = add(skip_connections)

		out = Activation('relu')(out)
		out = Conv1D(input_shape[-1], 1, padding='same'
			# kernel_constraint=maxnorm(2.)
			)(out)
		out = Activation('relu')(out)
		# out = Conv1D(input_shape[-1], 1, padding='same')(out) # original code
		out = Conv1D(1, 1, padding='same')(out)
		out = BatchNormalization()(out)

		if not learn_all_outputs:
			raise DeprecationWarning('Learning on just all outputs is wasteful, now learning only inside receptive field.')
			out = Lambda(lambda x: x[:, -1, :], output_shape=(out._keras_shape[-1],))(out)  # Based on gif in deepmind blog: take last output?

		out = Flatten()(out)
		# out = Activation('softmax', name="output_softmax")(out)

		out = Dense(16, 
			activation= 'linear',
			kernel_initializer= TruncatedNormal(mean=0.0, stddev=0.05),
			use_bias= True,
			bias_initializer= 'glorot_normal')(out)

		print(input_shape, input_shape[-1])
		out = Dense(output_classes, activation='softmax')(out)

		return input, out