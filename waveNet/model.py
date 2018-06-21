"""
The file to create the model. Reference: 
"""

import numpy as np
import tensorflow as tf

def create_variable(name, shape):
	'''Create a convolution filter variable with the specified name and shape,
	and initialize it using Xavier initialition.'''
	initializer = tf.contrib.layers.xavier_initializer_conv2d()
	variable = tf.Variable(initializer(shape=shape), name=name)
	# variable = tf.get_variable(name, shape = shape, initializer = initializer)
	return variable

def create_bias_variable(name, shape):
	'''Create a bias variable with the specified name and shape and initialize
	it to zero.'''
	initializer = tf.constant_initializer(value=0.0, dtype=tf.float32)
	bias = tf.Variable(initializer(shape=shape), name)
	return bias

def time_to_batch(value, dilation, name=None):
	with tf.name_scope('time_to_batch'):
		shape = tf.shape(value)
		pad_elements = dilation - 1 - (shape[1] + dilation - 1) % dilation
		padded = tf.pad(value, [[0, 0], [0, pad_elements], [0, 0]])
		reshaped = tf.reshape(padded, [-1, dilation, shape[2]])
		transposed = tf.transpose(reshaped, perm=[1, 0, 2])
		return tf.reshape(transposed, [shape[0] * dilation, -1, shape[2]])


def batch_to_time(value, dilation, name=None):
	with tf.name_scope('batch_to_time'):
		shape = tf.shape(value)
		prepared = tf.reshape(value, [dilation, -1, shape[2]])
		transposed = tf.transpose(prepared, perm=[1, 0, 2])
		return tf.reshape(transposed,
						  [tf.div(shape[0], dilation), -1, shape[2]])

def causal_conv(value, filter_, dilation, name='causal_conv'):
	with tf.name_scope(name):
		filter_width = tf.shape(filter_)[0]
		if dilation > 1:
			transformed = time_to_batch(value, dilation)
			conv = tf.nn.conv1d(transformed, filter_, stride=1,
								padding='VALID')
			restored = batch_to_time(conv, dilation)
		else:
			restored = tf.nn.conv1d(value, filter_, stride=1, padding='VALID')
		# Remove excess elements at the end.
		out_width = tf.shape(value)[1] - (filter_width - 1) * dilation
		result = tf.slice(restored,
						  [0, 0, 0],
						  [-1, out_width, -1])
		return result

class wavenet_model(object):
	"""
	Usage:
		dilations = [2**i for i in range(N)] * M
		filter_width = 2  # Convolutions just use 2 samples.
		residual_filters = 16  
		dilation_filters = 32  
		skip_filters = 16     
		net = WaveNetModel(batch_size, dilations, filter_width,
						   residual_filters, dilation_filters,
						   skip_filters)
		loss = net.loss(input_batch)

	Two major functions in this model is and _create_variables and _create_network.
	"""

	def __init__(self,
		batch_size,
		dilations,
		filter_width,
		residual_filters,
		dilation_filters,
		skip_filters,
		input_channels,
		output_classes,
		use_biases=False,
		global_condition_channels=None,
		global_condition_cardinality=None):
		"""
		Args:
			batch_size: How many audio files are supplied per batch
				(recommended: 1).
			dilations: A list with the dilation factor for each layer.
				e.g. [1,2,4]
			filter_width: The samples that are included in each convolution,
				after dilating.
			residual_filters: How many filters to learn for the residual.
			dilation_filters: How many filters to learn for the dilated
				convolution.
			skip_filters: How many filters to learn that contribute to the
				quantized softmax output.
			input_channels: How many sparse values to use for input 
				and the corresponding one-hot encoding.    
			output_classes: How many classes to output, as a classification problem
			use_biases: Whether to add a bias layer to each convolution.
				Default: False.
			# This model does not use scalar. one-hot only    
			# scalar_input: Whether to use the quantized waveform directly as
			#     input to the network instead of one-hot encoding it.
			#     Default: False.
			# initial_filter_width: The width of the initial filter of the
			#     convolution applied to the scalar input. This is only relevant
			#     if scalar_input=True.
			histograms: Whether to store histograms in the summary.
				Default: False.
			global_condition_channels: Number of channels in (embedding
				size) of global conditioning vector. None indicates there is
				no global conditioning.
			global_condition_cardinality: Number of mutually exclusive
				categories to be embedded in global condition embedding. If
				not None, then this implies that global_condition tensor
				specifies an integer selecting which of the N global condition
				categories, where N = global_condition_cardinality. If None,
				then the global_condition tensor is regarded as a vector which
				must have dimension global_condition_channels.
				"""


		self.batch_size = batch_size
		self.dilations = dilations
		self.filter_width = filter_width
		self.residual_filters = residual_filters
		self.dilation_filters = dilation_filters
		self.skip_filters = skip_filters
		self.input_channels = input_channels
		self.output_classes = output_classes
		self.use_biases = use_biases
		self.global_condition_channels = global_condition_channels
		self.global_condition_cardinality = global_condition_cardinality

		self.receptive_field = self.calculate_receptive_field(filter_width, dilations)
		self.variables = self._create_variables()

	# @staticmethod
	def calculate_receptive_field(self,filter_width, dilations):
		receptive_field = (filter_width - 1) * sum(dilations) + 1
		############ if dilation is [1,2], then 1*3+1 = 4
		############ and 4+1 = 5... this may be wrong
		# receptive_field += filter_width - 1 
		return receptive_field

	def _create_variables(self):
		'''This function creates all variables used by the network.
		This allows us to share them between multiple calls to the loss
		function and generation function.'''

		var = dict()

		with tf.variable_scope('wavenet'):

			with tf.variable_scope('causal_layer'):
				layer = dict()
				initial_channels = self.input_channels
				initial_filter_width = self.filter_width

				layer['filter'] = create_variable(
					'filter',
					[initial_filter_width,
					 initial_channels,
					 self.residual_filters])

				var['causal_layer'] = layer

		var['dilated_stack'] = list()
		with tf.variable_scope('dilated_stack'):
			for i, dilation in enumerate(self.dilations):
				with tf.variable_scope('layer{}'.format(i)):
					current = dict()
					current['filter'] = create_variable(
						'filter',
						[self.filter_width,
						 self.residual_filters,
						 self.dilation_filters])
					current['gate'] = create_variable(
						'gate',
						[self.filter_width,
						 self.residual_filters,
						 self.dilation_filters])
					current['dense'] = create_variable(
						'dense',
						[1,
						 self.dilation_filters,
						 self.residual_filters])
					current['skip'] = create_variable(
						'skip',
						[1,
						 self.dilation_filters,
						 self.skip_filters])

					if self.use_biases:
						current['filter_bias'] = create_bias_variable(
							'filter_bias',
							[self.dilation_filters])
						current['gate_bias'] = create_bias_variable(
							'gate_bias',
							[self.dilation_filters])
						current['dense_bias'] = create_bias_variable(
							'dense_bias',
							[self.residual_filters])
						current['skip_bias'] = create_bias_variable(
							'slip_bias',
							[self.skip_filters])

					var['dilated_stack'].append(current)

		with tf.variable_scope('postprocessing'):
			current = dict()
			current['postprocess1'] = create_variable(
				'postprocess1',
				[1, self.skip_filters, self.skip_filters])
			current['postprocess2'] = create_variable(
				'postprocess2',
				[1, self.skip_filters, self.output_classes])
			if self.use_biases:
				current['postprocess1_bias'] = create_bias_variable(
					'postprocess1_bias',
					[self.skip_filters])
				current['postprocess2_bias'] = create_bias_variable(
					'postprocess2_bias',
					[self.output_classes])
			var['postprocessing'] = current

		return var


	def _create_causal_layer(self, input_batch):
		'''Creates a single causal convolution layer.

		The layer can change the number of channels.
		'''
		with tf.name_scope('causal_layer'):
			weights_filter = self.variables['causal_layer']['filter']
			return causal_conv(input_batch, weights_filter, 1)

	def _create_dilation_layer(self, input_batch, layer_index, dilation,
							   global_condition_batch, output_width):
		'''Creates a single causal dilated convolution layer.

		Args:
			 input_batch: Input to the dilation layer.
			 layer_index: Integer indicating which layer this is.
			 dilation: Integer specifying the dilation size.
			 global_conditioning_batch: Tensor containing the global data upon
				 which the output is to be conditioned upon. Shape:
				 [batch size, 1, channels]. The 1 is for the axis
				 corresponding to time so that the result is broadcast to
				 all time steps.

		The layer contains a gated filter that connects to dense output
		and to a skip connection:

			   |-> [gate]   -|        |-> 1x1 conv -> skip output
			   |             |-> (*) -|
		input -|-> [filter] -|        |-> 1x1 conv -|
			   |                                    |-> (+) -> dense output
			   |------------------------------------|

		Where `[gate]` and `[filter]` are causal convolutions with a
		non-linear activation at the output. Biases and global conditioning
		are omitted due to the limits of ASCII art.

		'''
		variables = self.variables['dilated_stack'][layer_index]

		weights_filter = variables['filter']
		weights_gate = variables['gate']

		conv_filter = causal_conv(input_batch, weights_filter, dilation)
		conv_gate = causal_conv(input_batch, weights_gate, dilation)

		if self.use_biases:
			filter_bias = variables['filter_bias']
			gate_bias = variables['gate_bias']
			conv_filter = tf.add(conv_filter, filter_bias)
			conv_gate = tf.add(conv_gate, gate_bias)

		out = tf.multiply(tf.tanh(conv_filter), tf.sigmoid(conv_gate))

		# The 1x1 conv to produce the residual output
		weights_dense = variables['dense']
		transformed = tf.nn.conv1d(
			out, weights_dense, stride=1, padding="SAME", name="dense")

		# The 1x1 conv to produce the skip output
		skip_cut = tf.shape(out)[1] - output_width
		out_skip = tf.slice(out, [0, skip_cut, 0], [-1, -1, -1])
		weights_skip = variables['skip']
		skip_contribution = tf.nn.conv1d(
			out_skip, weights_skip, stride=1, padding="SAME", name="skip")

		if self.use_biases:
			dense_bias = variables['dense_bias']
			skip_bias = variables['skip_bias']
			transformed = transformed + dense_bias
			skip_contribution = skip_contribution + skip_bias

		input_cut = tf.shape(input_batch)[1] - tf.shape(transformed)[1]
		input_batch = tf.slice(input_batch, [0, input_cut, 0], [-1, -1, -1])

		return skip_contribution, input_batch + transformed

	def _generator_conv(self, input_batch, state_batch, weights):
		'''Perform convolution for a single convolutional processing step.'''
		# TODO generalize to filter_width > 2
		past_weights = weights[0, :, :]
		curr_weights = weights[1, :, :]
		output = tf.matmul(state_batch, past_weights) + tf.matmul(
			input_batch, curr_weights)
		return output

	def _generator_causal_layer(self, input_batch, state_batch):
		with tf.name_scope('causal_layer'):
			weights_filter = self.variables['causal_layer']['filter']
			output = self._generator_conv(
				input_batch, state_batch, weights_filter)
		return output

	def _generator_dilation_layer(self, input_batch, state_batch, layer_index,
								  dilation, global_condition_batch):
		variables = self.variables['dilated_stack'][layer_index]

		weights_filter = variables['filter']
		weights_gate = variables['gate']
		output_filter = self._generator_conv(
			input_batch, state_batch, weights_filter)
		output_gate = self._generator_conv(
			input_batch, state_batch, weights_gate)

		if self.use_biases:
			output_filter = output_filter + variables['filter_bias']
			output_gate = output_gate + variables['gate_bias']

		out = tf.tanh(output_filter) * tf.sigmoid(output_gate)

		weights_dense = variables['dense']
		transformed = tf.matmul(out, weights_dense[0, :, :])
		if self.use_biases:
			transformed = transformed + variables['dense_bias']

		weights_skip = variables['skip']
		skip_contribution = tf.matmul(out, weights_skip[0, :, :])
		if self.use_biases:
			skip_contribution = skip_contribution + variables['skip_bias']

		return skip_contribution, input_batch + transformed

	def _create_network(self, input_batch, global_condition_batch):
		'''Construct the WaveNet network.

		The layer contains a gated filter that connects to dense output
		and to a skip connection:

			   |                     |-> skip output --------------|
			   |    forward direction|                             |
		input -|-> [dilation stack] -|-> (+) -> dense output       |
			   |                                                   |-> (concatenate) -> (+) -> Relu
			   |------------------------------------|              |
																   |
			   |                     |-> skip output --------------|
			   |    backward direct  | 
		input -|-> [dilation stack] -|-> (+) -> dense output       
			   |                                    
			   |------------------------------------

		'''


		outputs = []
		current_layer = input_batch

		# Pre-process the input with a regular convolution
		initial_channels = self.input_channels

		current_layer = self._create_causal_layer(current_layer)

		output_width = tf.shape(input_batch)[1] - self.receptive_field 

		# Add all defined dilation layers.
		with tf.name_scope('dilated_stack'):
			for layer_index, dilation in enumerate(self.dilations):
				with tf.name_scope('layer{}'.format(layer_index)):
					output, current_layer = self._create_dilation_layer(
						current_layer, layer_index, dilation,
						global_condition_batch, output_width)
					outputs.append(output)

		"""
		add all opposite direction dilation layers
		"""

		"""
		postprocessing Method 1:
		Perform (+) -> (concatenate) -> ReLU -> 1x1 conv -> ReLU -> 1x1 conv
		
		postprocessing Method 2:
		Perform (concatenate) -> (+) -> ReLU -> 1x1 conv -> ReLU -> 1x1 conv
		"""

		with tf.name_scope('postprocessing'):
			"""
			concatenate two direction of dilated stack, then do 
			"""
			# Perform (+) -> ReLU -> 1x1 conv -> ReLU -> 1x1 conv to
			# postprocess the output.

			w1 = self.variables['postprocessing']['postprocess1']
			w2 = self.variables['postprocessing']['postprocess2']
			if self.use_biases:
				b1 = self.variables['postprocessing']['postprocess1_bias']
				b2 = self.variables['postprocessing']['postprocess2_bias']

			# if self.histograms:
			# 	tf.histogram_summary('postprocess1_weights', w1)
			# 	tf.histogram_summary('postprocess2_weights', w2)
			# 	if self.use_biases:
			# 		tf.histogram_summary('postprocess1_biases', b1)
			# 		tf.histogram_summary('postprocess2_biases', b2)

			# We skip connections from the outputs of each layer, adding them
			# all up here.
			total = sum(outputs)
			transformed1 = tf.nn.relu(total)
			conv1 = tf.nn.conv1d(transformed1, w1, stride=1, padding="SAME")
			if self.use_biases:
				conv1 = tf.add(conv1, b1)
			transformed2 = tf.nn.relu(conv1)
			conv2 = tf.nn.conv1d(transformed2, w2, stride=1, padding="SAME")
			if self.use_biases:
				conv2 = tf.add(conv2, b2)

		return conv2


	def predict_proba(self, input_batch, global_condition=None, name='wavenet'):
		with tf.name_scope(name):
			shape_ = [self.batch_size, -1, self.input_channels]
			encoded = tf.reshape(input_batch, shape_)
			# encoded = self._one_hot(waveform)

			raw_output = self._create_network(encoded, gc_embedding)
			out = tf.reshape(raw_output, [-1, self.output_classes])
			# Cast to float64 to avoid bug in TensorFlow
			proba = tf.cast(
				tf.nn.softmax(tf.cast(out, tf.float64)), tf.float32)
			last = tf.slice(
				proba,
				[tf.shape(proba)[0] - 1, 0],
				[1, self.output_classes])

			return tf.reshape(last, [-1])


	def loss(self,
			 input_batch,
			 target_output,
			 global_condition_batch=None,
			 l2_regularization_strength=None,
			 name='wavenet'):
		'''Creates a WaveNet network and returns the autoencoding loss.

		The variables are all scoped to the given name.
		'''
		with tf.name_scope(name):
			# encoded = self._one_hot(encoded_input)

			network_input = input_batch

			# Cut off the last sample of network input to preserve causality.
			# network_input_width = tf.shape(network_input)[1] - 1
			network_input_width = tf.shape(network_input)[1]
			network_input = tf.slice(network_input, [0, 0, 0],
									 [-1, network_input_width, -1])

			raw_output = self._create_network(network_input, global_condition_batch)

			with tf.name_scope('loss'):
				# Cut off the samples corresponding to the receptive field
				# for the first predicted sample.

				target_output = tf.reshape(target_output,
										   [-1, self.output_classes])
				prediction = tf.reshape(raw_output,
										[-1, self.output_classes])
				loss = tf.nn.softmax_cross_entropy_with_logits(
					logits=prediction,
					labels=target_output)
				reduced_loss = tf.reduce_mean(loss)

				tf.summary.scalar('loss', reduced_loss)

				if l2_regularization_strength is None:
					return reduced_loss
				else:
					# L2 regularization for all trainable parameters
					l2_loss = tf.add_n([tf.nn.l2_loss(v)
										for v in tf.trainable_variables()
										if not('bias' in v.name)])

					# Add the regularization term to the loss
					total_loss = (reduced_loss +
								  l2_regularization_strength * l2_loss)

					tf.summary.scalar('l2_loss', l2_loss)
					tf.summary.scalar('total_loss', total_loss)

					return total_loss