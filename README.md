# Udacity_Computer_Vision_Nanodegree

This repo contains my work for the Udacity Computer Vision Nanodegree Program.  Each directory contains a project required for completion and also contains instructions on what is expected to complete each project successfully.  Also note that the code uses PyTorch 1.x and not 0.x so some slight changes were made to ensure the code runs.

## Project #1 - Facial Keypoints Detection

The goal of this project is to detect where the facial keypoints are on a face image by designing an appropriate deep neural network to directly regress their locations on the face.  The relevant models, notebooks and results can be found in `Facial-Keypoint-Detection-P1`.   The trained model for this project has been committed via Git LFS so this will need to be installed if you want to grab the model.

Please visit <https://github.com/git-lfs/git-lfs/wiki/Installation> to install Git LFS. Once you do, after you clone this repo, please use the following to download the weights.

```sh
$ cd <path to this repo>
$ git lfs fetch
```

## Project #2 - Automatic Image Captioning

The goal of this project is to construct a CNN-RNN network that aims to automatically caption images.  The network is trained on the Microsoft COCO dataset which has ground truth captions for images in the dataset.  Please note that if you wish to run the notebooks to train the model and perform inference, you will need to download the 2014 train, validation and test images from COCO.  Please go here for more details:  http://cocodataset.org/#download.

You will also need to download the COCO API in order to navigate through the images and captions.  Please go to https://github.com/cocodataset/cocoapi for more information.  The notebooks required to go from start to finish can be found in `Image-Captioning-Project-P2`. 

There are notebook cells that run commands to download the dataset and the dependencies required to navigate through the images as well as installing the COCO API on your local machine for use.  These can be found in the `Image-Captioning-Project-P2/0_Dataset.ipynb` notebook.  It is imperative that you run this notebook first regardless of whether you are training or just testing out the algorithm.

Please note that the 2014 training, validation and testing images and annotations are roughly 40 GB of space in total, so please make sure you have this much space prior to downloading the dataset.  However, the cells incrementally download each portion of the dataset one zip file at a time by extracting the contents then deleting the zip file.  Therefore, you should run the `Image-Captioning-Project-P2/1_Preliminaries.ipynb` and `Image-Captioning-Project-P2/2_Training.ipynb` notebooks.

Should you not wish to train the model, the CNN encoder and RNN decoder models are stored in the `Image-Captioning-Project-P2/models` directory.  The `3_inference.ipynb` notebook file shows examples of how to run the captioning algorithm given an input image.