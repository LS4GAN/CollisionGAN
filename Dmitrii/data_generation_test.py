#!/usr/bin/env python
from subprocess import check_output
import pandas as pd
import numpy as np



dataset = '/hpcgpfs01/scratch/yhuang2/merged/'
label = 'windows'
script = '/sdcc/u/yhuang2/PROJs/GAN/collisionGAN/Dmitrii/sandbox-dmitrii/toytools/scripts/preprocess'
min_signal = 100
num_samples = 1000

cmd = f'{script} {dataset} {label} --min-signal {min_signal} -n {num_samples}'
print(cmd)
output = check_output(cmd, shell=True)

