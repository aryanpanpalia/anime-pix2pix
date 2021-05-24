# anime-pix2pix

## Data
The dataset is obtained from https://www.animecharactersdatabase.com using the programs in the data gathering folder. First, `scraper.py` downloads images from the site. Then `create_dataset.py` formats those downloaded images into the format used to train, detecting the faces and then concatenating the face with an image of the face run through the Canny edge detection algorithm.

## Training
To train the model from the dataset, run `main.py.` If saved models already exist in the saved_model_paths directory, it will load them up and resume training on them.

## Results
The results/logs while training can be seen in the log directory using tensorboard. The logs are made using the validation data.