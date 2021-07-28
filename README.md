# CollisionGAN
Collision image translation

Dear GC3D teammates, 

For a simplied CycleGAN implementation, there are only three places you need to look at
1. `cycleGan.ipynb` that has the implementaion of CycleGAN class, data handling, training options, and training;
2. `models/base_model.py` (base model for CycleGAN), `models/resnet.py` (for generator), `models/alexnet.py` (for discriminator)
  I didn't use the original discriminator because it is sensitive to dimension, and Dmitrii's AlexNet works very well.
3. `utils/network.py` that has a few utility functions.

I hack the original CycleGAN code and make it simply because
1. I want to force myself to look as close as possible into it which may make it easier if we want to utilize some part of there code;
2. The original implementation is very comprehensive but some parts of it is not very necessary for us and distractive;
3. The simplification help me to compartmentalize my understanding of the building blocks of CycleGAN.


CLAIM: I haven't give check it very my simplification a thorough check. 

TO-DO:
1. I will use the `PIL` Python package to convert our npz files to images, and run the original CycleGAN in order to check whether my simplification works properly;
2. Use paired test dataset to see how the translation works.
