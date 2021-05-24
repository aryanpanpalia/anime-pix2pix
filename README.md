# anime-pix2pix

## Data
The dataset is obtained from https://www.animecharactersdatabase.com using the programs in `data_gathering/`. First, `scraper.py` downloads images from the site. Then `create_dataset.py` formats those downloaded images into the format used to train, detecting the faces and then concatenating the face with an image of the face run through the Canny edge detection algorithm.

![image](https://github.com/aryanpanpalia/anime-pix2pix/blob/main/examples/data/image.png)

The image above is an example of an image in the dataset. The first half is the normal face and the second half is the face after being ran through the Canny edge detector.

## Training
To train the model from the dataset, run `main.py.` If saved models already exist in `saved_model_paths/`, it will load them up and resume training on them. Each hair color is trained and saved separately. The model is saved every 2000 steps to `saved_model_paths/.`

## Results
The results/logs while training can be seen in the log directory using tensorboard. The logs are made using the validation data. Some results for orange hair are below.

Condition (what the generator sees):
![image](https://github.com/aryanpanpalia/anime-pix2pix/blob/main/examples/results/orange_condition.png)

Generated (what the generator makes):
![image](https://github.com/aryanpanpalia/anime-pix2pix/blob/main/examples/results/orange_fake.png)

Real (what the generator was trying to make):
![image](https://github.com/aryanpanpalia/anime-pix2pix/blob/main/examples/results/orange_real.png)