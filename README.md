<h1 style="align: center">

ASL Alphabet Recognition

</h1>

<p style="align: justify">

This project aims to recognize American Sign Language (ASL) alphabet using deep learning techniques.

</p>

## Description

<p style="align: justify">

The project preprocesses images of hands showing ASL signs and uses Fastai models to classify them. <br>
It uses Streamlit for the web interface, Mediapipe to recognize hands, and OpenCV to crop the bounding box of the hand. <br>
The dataset used is the <a href="https://www.kaggle.com/datasets/grassknoted/asl-alphabet">ASL Alphabet dataset</a>.

</p>

## Model Architecture

### Model V1

<p style="justify">

**Architecture**:
  - Uses a pre-trained ResNet34 model from the Fastai library.
  - The final layers are customized to fit the number of ASL alphabet classes.
</p>

<p style="align: justify">

**Training**:
  - The dataset is split into training and validation sets using a random splitter with 20% validation data.
  - Data augmentation techniques are applied, for example: resizing images to 224x224 pixels.
  - The model is trained using the One Cycle Policy for 4 epochs.
  - Cross-entropy loss is used as the loss function.
  - The learning rate is fine-tuned using the learning rate finder.
  - The model's performance is evaluated using accuracy metrics on the validation set.
</p>

### Model V2

<p style="align: justify">

**Architecture**:
  - Uses a pre-trained ResNet50 model from the Fastai library.
  - The final layers are customized to fit the number of ASL alphabet classes.
</p>

<p style="align: justify">

**Training**:
  - The dataset is split into training and validation sets using a random splitter with 20% validation data.
  - Data augmentation techniques are applied, for example: resizing images to 224x224 pixels with padding, and various transformations like brightness, contrast, rotation, dihedral, and random resized crop.
  - The model is trained using the One Cycle Policy for 4 epochs with a base learning rate of 1e-3.
  - Cross-entropy loss is used as the loss function.
  - The learning rate is fine-tuned using the learning rate finder.
  - The model's performance is evaluated using accuracy metrics on the validation set.
</p>

## Screenshots

![img](./img/1.png)