# Face Features detection using Fastai and CelebA dataset.

CelebFaces Attributes Dataset (CelebA) is a large-scale face attributes dataset with more than 200K celebrity images, each with 40 attribute annotations. The images in this dataset cover large pose variations and background clutter. CelebA has large diversities, large quantities, and rich annotations, including

10,177 number of identities,

202,599 number of face images, and

5 landmark locations, 40 binary attributes annotations per image.

The dataset can be employed as the training and test sets for the following computer vision tasks: face attribute recognition, face detection, landmark (or facial part) localization, and face editing & synthesis.
This data was originally collected by researchers at MMLAB, The Chinese University of Hong Kong .


# Training the dataset.

For this dataset i used the architecture resnet50 to train my model with 1 epoch. The accuracy threshold was between 85% to 90% depending on the threshold.

![alt text](https://github.com/Miske1996/Face-Features/blob/master/resnet50.jpg)

# Testing my model.
For testing the model i downloaded some some images from the internet to see what the model will predict. 

![Output GIF](https://github.com/Miske1996/Face-Features/blob/master/test_model.gif)

# Dependencies
- fastai
- Pytorch
- opencv
- numpy
