#!/usr/bin/env python
import torch
import torch.nn as nn

class AlexNetDiscriminator(nn.Module):
    def __init__(self, input_channels=3, **kwargs):
        super(AlexNetDiscriminator, self).__init__(**kwargs)
        self.base_net = nn.Sequential(
            nn.BatchNorm2d(input_channels),

            nn.Conv2d(input_channels, 96, kernel_size=5),
            nn.ReLU(),
            nn.AvgPool2d(2, stride=2),

            nn.Conv2d(96, 256, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.AvgPool2d(2, stride=2),

            nn.Conv2d(256, 384, kernel_size=3, padding=1),
            nn.ReLU(),

            nn.Conv2d(384, 384, kernel_size=3, padding=1),
            nn.ReLU(),

            nn.Conv2d(384, 384, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        self.classifier = nn.Linear(384, 1)

    def forward(self, x):
        y = self.base_net(x)
        z = torch.mean(y, dim=(2, 3), keepdim=False) # Global average
        return self.classifier(z)
