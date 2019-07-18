# cgan_deeplearninglab
Implementing a conditional GAN to create faces from attributes and landmarks,
using the celeba dataset http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html

Based on arxiv.org/abs/1611.07004 (Image-to-Image Translation with Conditional Adversarial Networks)

Usage:
Download dataset (all pictures, list_attr_celeba.txt, list_landmarks_align_celeba.txt)
Run gpu_training.py or gpu_training_dropout.py to train new models
Run test.py or test_dropout.py to test the pre-trained models and to generate new images

The normal variant uses a random noise vector to introduce variance, 
the _dropout variant uses dropout-layers as suggest in the research paper.
