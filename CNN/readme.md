# CNN Guide

### Model

* Scratch CNN: pure hand build CNN with 55% accuracy on frame generation dataset(our old data)

* Resnet 18: ResNet18 is a variant of the Residual Network (ResNet) architecture, which was introduced to address the vanishing gradient problem in deep neural networks. The architecture is designed to allow networks to be deeper, thus improving their ability to learn complex patterns in data. 

    * Using pytorch to load resnet

    * To download and run, do:

        * Download data from [Kaggle](https://www.kaggle.com/datasets/manjilkarki/deepfake-and-real-images)

        * Run `train.py`, do not run `train_raw.py` will take a VERY long time and make any local computer out of memory.

        * To accelerate, consider upload `data.zip`, `train.py`, and `submit-train.sh` to submit job on Ocscar.

        * Each epoch takes around 35-40 minutes of batch size 256, 512.

        * Train log and output is listed in the folder.
    
    * 87% accuracy on Kaggle data

### XAI

* `visual.py` generates Grad-CAM

* `lime_visual.py` generates LIME

* `scorecam.py` generates Score-CAM(another form of Grad-CAM)

* `captum.py` generates captum integrated gradient.

* Further details can be found in report.