
import os
import cv2
import keras.backend as K
from utils import load_model
import numpy as np
import imgaug as ia
from imgaug import augmenters as iaa


def check_image_with_pil(path):
    extensions = {".jpg", ".png"}
    return any(path.endswith(ext) for ext in extensions)


def load_convnet():
    model = load_model()
    model.summary()
    return model

def check_is_npz(path):
    extensions = {".npz"}
    return any(path.endswith(ext) for ext in extensions)

def delete_npz(paths):
    for path in paths:
        if check_is_npz(path):
            os.remove(path)

def find_paths(path):
    paths = []
    level_a = os.listdir(path)
    for level_name in level_a:
        if not os.path.isdir(os.path.join(path, level_name)):
            continue
        for image_name in os.listdir(os.path.join(path, level_name)):
            img_path = os.path.join(path, level_name, image_name)
            if check_image_with_pil(img_path):
                paths += [img_path]
    return paths


def load_image_data(path):
    im_data = cv2.imread(path)
    # if im_data.shape[0] < im_data.shape[1]:
    #     im_data = cv2.transpose(im_data)
    #     im_data = cv2.flip(im_data, flipCode=1)
    im_data = cv2.resize(im_data, (224, 224))
    im_data = im_data / 255.
    return im_data


def main():
    net = load_convnet()
    get_features = K.function([net.layers[0].input, K.learning_phase()], [
                              net.get_layer("flatten_2").output])

    base_path = "/Users/carlosleao/workspace/ml-deeplearning/datasets/DeepLearningFiles/"
    paths = find_paths(base_path)
    for image_path in paths:
        print image_path

        im_data = load_image_data(image_path)
        augmentated_list_image = augmentate_image(im_data)
        for i in range(len(augmentated_list_image)):
            features = get_features(
                [augmentated_list_image[i][np.newaxis, ...], 0])[0]
            print features.shape
            # cv2.imshow("frameB",im_data)
            npz_path = "{}-{}.npz".format(image_path, i)
            print "NPZ Path" + npz_path
            np.savez_compressed(npz_path, **{"f1": features})
        # cv2.waitKey(0)


def augmentate_image(im_data):
    base_image = im_data.copy()
    im_augmented = [im_data]
    # Rotate the image in all directions
    # for i in range(3):
    #     h, w = base_image.shape[:2]
    #     center = (w / 2, h / 2)
    #     M = cv2.getRotationMatrix2D(center, 90, 1.0)
    #     base_image = cv2.warpAffine(base_image, M, (h, w))
    #     # dst = cv2.flip(base_image, flipCode=1)
    #     im_augmented += [base_image.copy()]


    return im_augmented


if __name__ == "__main__":
    main()
