import torch
from torch import nn

def get_layer_depth(layer, depth_seed, growth):
	if growth == 'linear':
		return (1 + layer) * depth_seed

	if growth == 'exponential':
		return 2**layer * depth_seed

	if growth == 'constant':
		return depth_seed

	raise ValueError("Unknown growth type: '%s'" % growth)


def get_activ_layer(activation):
	if activation == 'relu':
		return nn.ReLU()

	if activation == 'selu':
		return nn.SELU()

	raise ValueError("Unknown activation: '%s'" % activation)


def unet_block(
	input_channels, 
	output_channels, 
	activation, 
	**conv_kwargs
):
	
	layer_activ = get_activ_layer(activation)

	return nn.Sequential(
		nn.BatchNorm2d(input_channels),
		nn.Conv2d(
			input_channels, 
			output_channels, 
			**conv_kwargs
		),
		layer_activ,
		nn.BatchNorm2d(output_channels),
		nn.Conv2d(
			output_channels, 
			output_channels, 
			**conv_kwargs
		),
		layer_activ,
	)


def construct_base_encoder(
	input_channels, 
	depth_seed, 
	growth, 
	activation
):
	
	output_channels = get_layer_depth(0, depth_seed, growth)

	return nn.Sequential(
		# adjusting
		unet_block(
			input_channels, 
			output_channels, 
			activation,
			kernel_size=3, 
			padding=1
		),
		# downsampling
		nn.AvgPool2d(2, stride=2),
	)


def construct_base_decoder(
	input_channels, 
	depth_seed, 
	growth, 
	activation
):

	output_channels = get_layer_depth(0, depth_seed, growth)

	return nn.Sequential(
		# upsampling
		nn.ConvTranspose2d(
			output_channels, 
			output_channels, 
			kernel_size=2, stride=2
		),
		# adjusting
		unet_block(
			output_channels, 
			output_channels, 
			activation,
			kernel_size=3, 
			padding=1
		),
		# further adjusting
		nn.Conv2d(output_channels, input_channels, kernel_size=1)
	)


def add_encoder_blocks(
	encoder, 
	depth_seed, 
	blocks, 
	growth, 
	activation
):

	for block_idx in range(blocks):
		input_channels  = get_layer_depth(0 + block_idx, depth_seed, growth)
		output_channels = get_layer_depth(1 + block_idx, depth_seed, growth)

		encoder.add_module(
			'enc_block_%d' % block_idx,
			nn.Sequential(
				unet_block(
					input_channels, 
					output_channels, 
					activation,
					kernel_size=3, 
					padding=1 
				),
				nn.AvgPool2d(2, stride = 2),
			)
		)


def add_decoder_blocks(
	decoder, 
	depth_seed, 
	blocks, 
	growth, 
	activation
):

	for block_idx in reversed(range(blocks)):
		input_channels  = get_layer_depth(0 + block_idx, depth_seed, growth)
		output_channels = get_layer_depth(1 + block_idx, depth_seed, growth)

		decoder.add_module(
			'dec_block_%d' % block_idx,
			nn.Sequential(
				nn.ConvTranspose2d(
					output_channels, 
					output_channels,
					kernel_size=2, 
					stride=2
				),
				unet_block(
					output_channels, 
					input_channels, 
					activation,
					kernel_size=3, 
					padding=1
				),
			)
		)


def construct_encoder(
	input_channels, 
	depth_seed, 
	blocks, 
	growth, 
	activation
):
	
	result = nn.Sequential()

	result.add_module(
		'enc_base',
		construct_base_encoder(
			input_channels, 
			depth_seed, 
			growth, 
			activation
		)
	)

	add_encoder_blocks(
		result, 
		depth_seed, 
		blocks, 
		growth, 
		activation
	)

	return result


def construct_bottleneck(
	input_channels, 
	depth_seed, 
	blocks, 
	growth, 
	activation
):
	outer_channels = get_layer_depth(blocks + 0, depth_seed, growth)
	inner_channels = get_layer_depth(blocks + 1, depth_seed, growth)
	layer_activ	= get_activ_layer(activation)

	return nn.Sequential(
		nn.BatchNorm2d(outer_channels),
		nn.Conv2d(
			outer_channels, 
			inner_channels, 
			kernel_size=3, 
			padding=1
		),
		layer_activ,
		nn.BatchNorm2d(inner_channels),
		nn.Conv2d(
			inner_channels, 
			outer_channels, 
			kernel_size=3, 
			padding=1
		),
		layer_activ,
	)


def construct_decoder(
	input_channels, 
	depth_seed, 
	blocks, 
	growth, 
	activation
):
	result = nn.Sequential()
	add_decoder_blocks(
		result, 
		depth_seed, 
		blocks, 
		growth, 
		activation
	)

	result.add_module(
		'dec_base',
		construct_base_decoder(
			input_channels, 
			depth_seed, 
			growth, 
			activation
		)
	)

	return result


class NotUNet(nn.Module):

	def __init__(
		self, 
		input_channels=1, 
		depth_seed=16, 
		blocks=2,
		growth='linear', 
		activation='relu', 
		**kwargs
	):
		super().__init__(**kwargs)

		self.encoder = construct_encoder(
			input_channels, 
			depth_seed, 
			blocks, 
			growth, 
			activation
		)

		self.bottleneck = construct_bottleneck(
			input_channels, 
			depth_seed, 
			blocks, 
			growth, 
			activation
		)

		self.decoder = construct_decoder(
			input_channels, 
			depth_seed, 
			blocks, 
			growth, 
			activation
		)

	def get_transferable_params(self):
		result = (self.encoder.state_dict(), self.decoder.state_dict())
		return result

	def load_transferable_params(self, params):
		self.encoder.load_state_dict(params[0], strict=False)
		self.decoder.load_state_dict(params[1], strict=False)

	def make_lower_subnetwork_trainable(self, trainable=True):
		# I don't really understand this part
		# all trainable except the last
		for idx in range(len(self.encoder) - 1):
			layer = self.encoder[idx]
			for param in layer.parameters():
				param.requires_grad = trainable
		# all trainable except the first
		for idx in range(1, len(self.decoder)):
			layer = self.decoder[idx]
			for param in layer.parameters():
				param.requires_grad = trainable

	def encode(self, x):
		return self.bottleneck(self.encoder(x))

	def decode(self, z):
		return self.decoder(z)

	def forward(self, x):
		z = self.encode(x)
		return self.decode(z)
