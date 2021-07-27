#!/usr/bin/env python
import torch
import torch.nn as nn


def build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias):
    """
    Construct a convolutional block
    Parameters:
        dim (int) -- the number of channels in the conv layer.
        padding_type (str) -- the type of padding layer: reflect | replicate | zero
        norm_layer -- normalization layer
        use_dropout (bool) -- whether to use dropout layers
        use_bias (bool) -- whether use bias in the conv layers

    Return a conv block (with a conv layer, a normalization layer, and a non-lineary layer(ReLU))
    """
    conv_block = []
    p = 0
    if padding_type == 'reflect':
        conv_block.append(nn.ReflectionPad2d(1))
    elif padding_type == 'replicate':
        conv_block.append(nn.ReplicationPad2d(1))
    elif padding_type == 'zero':
        p = 1
    else:
        raise NotImplementedError(f'padding {padding_type} is not implemented')

    conv_block += [
        nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
        norm_layer(dim),
        nn.ReLU(True)
    ]
    if use_dropout:
        conv_block.append(nn.Dropout(.5))

    p = 0
    if padding_type == 'reflect':
        conv_block.append(nn.ReflectionPad2d(1))
    elif padding_type == 'replicate':
        conv_block.append(nn.ReplicationPad2d(1))
    elif padding_type == 'zero':
        p = 1
    else:
        raise NotImplementedError(f'padding {padding_type} is not implemented')

    conv_block += [
        nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
        norm_layer(dim)
    ]

    return nn.Sequential(*conv_block)


class ResnetBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """
        Initialize the Resnet block
        
        A resnet block is a Conv block with skip connections.
        We construct a Conv block with build_conv_block function,
        and implement skip connections in <forward> functions.
        """
        super(ResnetBlock, self).__init__()
        self.conv_block = build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def forward(self, x):
        return x + self.conv_block(x)


class ResnetGenerator(nn.Module):
    """
    Resnet-based generator that consists of Resnet blocks 
    between a frew downsampling/upsampling operation.
    We adapt Torch code and idea from Justin Johnson's neural style transfer project
    (https://github.com/jcjohnson/fast-neural-style).
    
    Width and height of size 4k are preserved.
    Width and height of size 4k + i, i = 1, 2, 3, will be mapped to 4(k + 1).  
    """
    
    def __init__(
        self, 
        input_nc, 
        output_nc,
        ngf=64,
        norm_layer=nn.BatchNorm2d,
        n_blocks=6, 
        padding_type='reflect',
        use_dropout=False
    ):
        """
        Construct a Resnet-based Generator
        Parameters:
            input_nc (int) -- the number of channels in the input
            output_nc (int) -- the number of channels in the output
            ngf (int) -- the number of filters in the leading conv layer
            norm_layer (torch layer) -- normalization layer
            n_blocks (int) -- the number of ResNet blocks
            padding_type (str) -- the type of padding layer in conv layers: reflect | replicate | zero
            use_dropout (bool) -- whether use dropout layers in the ResNet blocks
        """
        
        assert(n_blocks > 0), "n_blocks must be greater than zero!"
        super(ResnetGenerator, self).__init__()
        use_bias = (norm_layer == nn.InstanceNorm2d)
        
        # Add leading layers
        model = [
            nn.ReflectionPad2d(3), # padding=3
            nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
            norm_layer(ngf),
            nn.ReLU(True) # Inplace
        ]
        
        # Add downsampling layers
        n_downsamplings = 2
        mult = 1
        for i in range(n_downsamplings):
            ic, oc = ngf * mult, ngf * mult * 2
            mult *= 2
            model += [
                # @Yi pay attention to this padding
                nn.Conv2d(ic, oc, kernel_size=3, stride=2, padding=1, bias=use_bias),
                norm_layer(oc),
                nn.ReLU(True)
            ]
        
        # Add ResNet blocks
        for i in range(n_blocks):
            model.append(
                ResnetBlock(
                    ngf * mult, 
                    padding_type=padding_type, 
                    norm_layer=norm_layer, 
                    use_dropout=use_dropout,
                    use_bias=use_bias
                )
            )
            
        # Add upsampling layers
        for i in range(n_downsamplings):
            ic, oc = ngf * mult, ngf * mult // 2
            mult //= 2
            model += [
                # @Yi pay attention to the padding and output_padding here
                nn.ConvTranspose2d(
                    ic, oc, 
                    kernel_size=3, stride=2, 
                    padding=1, output_padding=1, 
                    bias=use_bias),
                norm_layer(oc),
                nn.ReLU(True)
            ]
        
        model.append(nn.ReflectionPad2d(3))
        model.append(nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0))
        model.append(nn.Tanh())
        
        self.model = nn.Sequential(*model)
    
    def forward(self, x):
        return self.model(x)

if __name__ == '__main__':
	# The following trial shows that
	# 1. Width and height of size $4k$ are preserved.
	# 2. Width and height of size $4k + i$, $i = 1, 2, 3$, will be mapped to $4(k + 1)$.  
	
	for w in range(160, 210):
	    input_shape = [w, w]
	    x = torch.rand(1, 1, *input_shape, dtype=torch.float32)
	    resnet = ResnetGenerator(1, 1)
	    
	    shape = resnet(x).shape[3]
	    print(f'{w}: {shape}')

