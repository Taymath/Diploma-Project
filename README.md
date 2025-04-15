For the program to work correctly, you need to download the MS-COCO 2017 dataset. Stable Diffusion v1.5 installed on your PC

Also, since it is assumed that the problem is solved at low power (in my case, I use a mobile video card Nvidia Geforce RTX 3060 with 6 GB VRAM),

the network is trained on images of size 64x64, with a sufficiently large number of epochs and a truncated dataset (in this version, 32,000 images are taken with batch_size = 32)

To stabilize learning, the following are used: ReduceLROnPlateau, EMATracker, 

To speed up calculations and reduce memory consumption: AMP

The FID method is used to compare the quality of the generated images and the final assessment.
