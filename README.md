# anime-pix2pix

## Prerequisites
* An operating system that can run python 3
* Python 3 (downloadable at https://www.python.org/downloads/)
* Enough spare storage space to hold the python packages and the dataset (at least 10 GB recommended)

## Setup
First, download the repository in any way you would see fit. Then run `pip install -r requirements.txt` to install the needed python packages.

## Data
The dataset is obtained from https://www.animecharactersdatabase.com using the programs in `data_gathering/`. First, `scraper.py` downloads images from the site. Then `create_dataset.py` formats those downloaded images into the format used to train, detecting the faces and then concatenating the face with an image of the face run through the Canny edge detection algorithm.

![image](https://github.com/aryanpanpalia/anime-pix2pix/blob/main/examples/data/image.png)

The image above is an example of an image in the dataset. The first half is the normal face and the second half is the face after being ran through the Canny edge detector.

## Training
It is recommended to use an NVIDIA GPU to train the model, however, if one is not available it will automatically train on CPU. To train the model from the dataset, run `main.py.` If saved models already exist in `saved_model_paths/`, it will load them up and resume training on them. Each hair color is trained and saved separately. The model is saved every 2000 steps to `saved_model_paths/.` 

## Results
The results/logs while training can be seen in the log directory using tensorboard. To see the logs, make sure to have tensorboard installed and go into the project directory and run `tensorboard --logdir logs.` The logs are made using the validation data. Some results for orange hair after 8000 steps are below.

Condition (what the generator sees):
![image](https://github.com/aryanpanpalia/anime-pix2pix/blob/main/examples/results/orange_condition.png)

Generated (what the generator makes):
![image](https://github.com/aryanpanpalia/anime-pix2pix/blob/main/examples/results/orange_fake.png)

Real (what the generator was trying to make):
![image](https://github.com/aryanpanpalia/anime-pix2pix/blob/main/examples/results/orange_real.png)
