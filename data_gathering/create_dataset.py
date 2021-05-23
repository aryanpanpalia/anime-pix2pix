import concurrent.futures
import os
import sys
import warnings

import cv2
import numpy as np
from tqdm import tqdm

warnings.filterwarnings(
    'ignore',
    message='libpng warning: iCCP: known incorrect sRGB profile'
)


def detect(filename, outname):
    cascade = cv2.CascadeClassifier("./cascade.xml")
    image = cv2.imread(filename, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)

    faces = cascade.detectMultiScale(gray, minNeighbors=5)

    for face_num in range(len(faces)):
        x, y, w, h = faces[face_num]

        side_len = w
        buffer = side_len * 0.4

        face_img = image[max(0, int(y - buffer)):int(y + h + buffer), max(0, int(x - buffer)):int(x + w + buffer)]
        face_img = cv2.resize(face_img, dsize=(256, 256))

        edges = ~cv2.Canny(face_img, 75, 45, L2gradient=True)
        edges = edges.reshape((256, 256, 1))
        edges = np.tile(edges, (1, 1, 3))

        concat = np.concatenate((face_img, edges), axis=1)

        cv2.imwrite(
            filename=f"{outname}_{face_num}.png",
            img=concat
        )


if __name__ == '__main__':
    IMAGE_DIR_PATH = './images'
    DATASET_DIR_PATH = '../dataset'

    os.mkdir(DATASET_DIR_PATH)

    for hair_color in ['black', 'blonde', 'blue', 'brown', 'green', 'grey', 'orange', 'pink', 'purple', 'red']:
        os.mkdir(f'{DATASET_DIR_PATH}/{hair_color}')

        image_names = os.listdir(f'{IMAGE_DIR_PATH}/{hair_color}')

        in_filenames = [f'{IMAGE_DIR_PATH}/{hair_color}/{image_name}' for image_name in image_names]
        out_filenames = [f'{DATASET_DIR_PATH}/{hair_color}/{image_name[0:-4]}' for image_name in image_names]

        with concurrent.futures.ThreadPoolExecutor() as executor:
            list(
                tqdm(
                    executor.map(detect, in_filenames, out_filenames),
                    total=len(in_filenames),
                    file=sys.stdout,
                    desc=f"Images processed (Hair color: {hair_color})"
                )
            )
