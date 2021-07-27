"""
yhuang2@bnl.gov
"""

import torch
import torch.nn as nn

import sys

from pathlib import Path
util_path = '/sdcc/u/yhuang2/PROJs/EICLDRD/'
assert Path(util_path).exists()
if util_path not in sys.path:
	sys.path.append(util_path)
# printing with style
from utils.text_style import text_style
ts = text_style()
# a simple stdout logger
from utils.dev_log import dev_log
logger = dev_log() 

# The Encoder
def get_encoder_block(input_channels, output_channels, kernel_size=1, stride=1, padding=0):
	return nn.Sequential(
		nn.Conv2d(input_channels, output_channels, kernel_size=kernel_size, stride=stride, padding=padding),
		nn.LeakyReLU(negative_slope=0.1),
		nn.InstanceNorm2d(num_features=output_channels)
	)

def get_encoder_block_double(input_channels, output_channels, kernel_size=1, stride=1, padding=0):
	return nn.Sequential(
		nn.Conv2d(input_channels, output_channels, kernel_size=kernel_size, stride=stride, padding=padding),
		nn.LeakyReLU(negative_slope=0.1),
		nn.InstanceNorm2d(num_features=output_channels),
		nn.Conv2d(output_channels, output_channels, kernel_size=3, stride=1, padding=1)
	)

class CNNEncoder(nn.Module):
	def __init__(self, csz, filters, strides, oneOneConvCsz=None, residual=True):
		"""
		A CNN auto-encoder that accepts variable channel sizes, filters, strides, and inclusion of a one-by-one convolutional layer.
		Use one-by-one convolutional layer for compression.
		"""
		super(CNNEncoder, self).__init__()
		
		self.residual = residual
		self.oneOneConvCsz = oneOneConvCsz
		
		assert len(csz) == len(filters) and len(csz) == len(strides), \
			"list of channels, filters, and strides must all have the same length!"

		# paddings = [[f // 2 for f in filter] for filter in filters]
		paddings = [[1 for f in filter] for filter in filters]
		
		csz_ = [1] + csz

		self.layers = nn.ModuleList()
		for i in range(len(csz)):
			layer = get_encoder_block(csz_[i], csz_[i + 1], kernel_size=filters[i], stride=strides[i], padding=paddings[i])
			self.layers.append(layer)
		
		if residual:
			self.double_layers, self.activation_layers = nn.ModuleList(), nn.ModuleList()
			for i in range(len(csz)):
				layer_double = get_encoder_block_double(csz_[i], csz_[i + 1], kernel_size=filters[i], stride=strides[i], padding=paddings[i])
				self.double_layers.append(layer_double)
				self.activation_layers.append(
					nn.Sequential(
						nn.LeakyReLU(negative_slope=0.1),
						nn.InstanceNorm2d(num_features=csz_[i + 1])
					)
				)
			
		if self.oneOneConvCsz:
			layer = get_encoder_block(csz_[-1], self.oneOneConvCsz)
			self.layers.append(layer)
			

	def forward(self, x):
		original_dim, original_size = x.shape, x.numel()
		
		if not self.residual:
			for layer in self.layers:
				y = layer(x)
				x = y
		else:
			# Torch scripting cannot handle `for i in range(len(self.layers))`
			# Torch scripting can handle enumerate and zip.
			for dlayer, slayer, alayer in zip(self.double_layers, self.layers, self.activation_layers):
				y = alayer(dlayer(x) + slayer(x))
				x = y
		
			if self.oneOneConvCsz:
				layer = self.layers[-1] 
				y = layer(x)
				x = y

		return x
	

# The Decoder
def get_decoder_block(input_channels, output_channels, kernel_size=1, stride=1, padding=0, output_padding=0, last=False, classification=True, linear=True):
	if last:
		if classification:
			return nn.Sequential(
				nn.ConvTranspose2d(input_channels, output_channels, kernel_size=kernel_size, stride=stride, padding=padding, output_padding=output_padding),
				nn.Sigmoid()
			)
		else:
			if linear:
				return nn.Sequential(
					nn.ConvTranspose2d(input_channels, output_channels, kernel_size=kernel_size, stride=stride, padding=padding, output_padding=output_padding),
				) 
			else:
				return nn.Sequential(
					nn.ConvTranspose2d(input_channels, output_channels, kernel_size=kernel_size, stride=stride, padding=padding, output_padding=output_padding),
					nn.ReLU() 
				) 
	else:
		if classification:
			return nn.Sequential(
				nn.ConvTranspose2d(input_channels, output_channels, kernel_size=kernel_size, stride=stride, padding=padding, output_padding=output_padding),
				nn.LeakyReLU(negative_slope=0.1),
				nn.InstanceNorm2d(num_features=output_channels)
			)
		else:
			return nn.Sequential(
				nn.ConvTranspose2d(input_channels, output_channels, kernel_size=kernel_size, stride=stride, padding=padding, output_padding=output_padding),
				nn.LeakyReLU(negative_slope=0.1),
				nn.InstanceNorm2d(num_features=output_channels)
			)
		

def get_decoder_block_double(input_channels, output_channels, kernel_size=1, stride=1, padding=0, output_padding=0):
	return nn.Sequential(
		nn.ConvTranspose2d(input_channels, output_channels, kernel_size=kernel_size, stride=stride, padding=padding, output_padding=output_padding),
		nn.LeakyReLU(negative_slope=0.1),
		nn.InstanceNorm2d(num_features=output_channels),
		# nn.ConvTranspose2d(output_channels, output_channels, kernel_size=3, stride=1, padding=1, output_padding=output_padding),
		nn.ConvTranspose2d(output_channels, output_channels, kernel_size=3, stride=1, padding=1, output_padding=0),
	)
		

class CNNDecoder(nn.Module):
	def __init__(self, csz, filters, strides, output_paddings, residual=True, evaluation_mode=False, classification=True, linear=True):
		super(CNNDecoder, self).__init__()
		self.residual = residual
		
		assert len(csz) == len(filters) and len(csz) == len(strides) and len(csz) == len(output_paddings), \
			"list of channels, filters, strides, and output_paddings must all have the same length!"

		self.evaluation_mode = evaluation_mode
		paddings = [[f // 2 for f in filter] for filter in filters]
		
		csz_ = csz + [1]
		
		self.layers = nn.ModuleList()
		for i in range(len(csz)):
			if i < len(csz) - 1:
				layer = get_decoder_block(csz_[i], csz_[i + 1], kernel_size=filters[i], stride=strides[i], padding=paddings[i], output_padding=output_paddings[i])
			else:
				layer = get_decoder_block(csz_[i], csz_[i + 1], kernel_size=filters[i], stride=strides[i], padding=paddings[i], output_padding=output_paddings[i], last=True, classification=classification, linear=linear)
			self.layers.append(layer)
			
		if residual:
			self.double_layers, self.activation_layers = nn.ModuleList(), nn.ModuleList()
			for i in range(len(csz) - 1):
				double_layer = get_decoder_block_double(csz_[i], csz_[i + 1], kernel_size=filters[i], stride=strides[i], padding=paddings[i], output_padding=output_paddings[i])
				self.double_layers.append(double_layer)
				self.activation_layers.append(
					nn.Sequential(
						nn.LeakyReLU(negative_slope=0.1),
						nn.InstanceNorm2d(num_features=csz_[i + 1])
					)
				)
				
	def forward_(self, x):
		if not self.residual:
			for layer in self.layers:
				y = layer(x)
				x = y
		else:
			# Torch scripting cannot handle `for i in range(len(self.layers))`
			# Torch scripting can handle enumerate and zip.
			for dlayer, slayer, alayer in zip(self.double_layers, self.layers, self.activation_layers):
				y = alayer(dlayer(x) + slayer(x))
				x = y
				
			layer = self.layers[-1] 
			y = layer(x)
			x = y
			
		return x

	def forward(self, x):
		if self.evaluation_mode:
			x = x.type(torch.float16)
			with torch.cuda.amp.autocast():
				return self.forward_(x)
		else:
			return self.forward_(x)
		
		return x
	

# The Auto-Encoder
class CNNAE(nn.Module):
	"""
	A double-headed auto-encoder
	The first output is the classification output and the second output is the regression output
	The regression output may be a tranform of the target value, hence adjust the decoder coder accordingly.
	"""
	def __init__(self, 
		csz, rcsz, 
		filters, strides, output_paddings, 
		oneOneConvCsz, *,
		residual=True,
		evaluation_mode=False, 
		single=False,
		linear=True
	):
		# super(CNNAE, self).__init__()
		super().__init__()
		
		assert (oneOneConvCsz is None and rcsz[0] == csz[-1]) or (rcsz[0] == oneOneConvCsz), \
			'The first input channel of decoder must be the same as either the one-by-one convolution channel of the encoder or the output channel of the last convolutional layer of the encoder.' 
# 		assert (single == False) or (single == True and linear == False), \
# 			'If single is true, linear must be false'
		
		self.single = single
		
		self.evaluation_mode = evaluation_mode
		self.encoder = CNNEncoder(csz, filters, strides, oneOneConvCsz=oneOneConvCsz, residual=residual)
		self.decoder_r = CNNDecoder(
			rcsz, 
			filters[::-1], strides[::-1], output_paddings[::-1], 
			linear=linear,
			residual=residual, 
			evaluation_mode=self.evaluation_mode, 
			classification=False)  

		if self.single == False:
			self.decoder_c = CNNDecoder(
				rcsz, 
				filters[::-1], strides[::-1], output_paddings[::-1], 
				residual=residual, 
				evaluation_mode=self.evaluation_mode, 
				classification=True
			)

	def forward(self, x):
		y = self.encoder(x)
        print(y.shape)
		logger.debug(f'output shape = {y.shape}\n')
		compression_ratio = torch.prod(torch.tensor(x.shape)) / torch.prod(torch.tensor(y.shape))
		compression_ratio = compression_ratio.cpu().detach().numpy()
		logger.debug(f'compression ratio = {compression_ratio: .3f}')
		y_r = self.decoder_r(y)

		if self.single == False:
			y_c = self.decoder_c(y)
			return y_c, y_r
		else:
			return y_r
