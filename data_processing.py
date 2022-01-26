import json
from re import X
from tkinter import W
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from PIL import Image
from glob import glob

def SingleFolderProcess(path, result, limit=150):
    x = []
    y = []
    
    os.chdir(path)
    img_list = glob("*.png")

    for i, img in enumerate(img_list[0:limit]):
        image = cv2.imread(path + img, cv2.IMREAD_COLOR)
        image = cv2.resize(image, dsize=(224, 224))

        arr = np.array(image).astype(float)/255
        x.append(arr)
        y.append(result)

        if (i%100==0):
            print("PROCESSING", i, "/", len(img_list))

    return x, y

def DataProcess(path, limit=150):
    print("PROCESS DOWN ...")
    downX, downY = SingleFolderProcess(path + "/DOWN/", [0, 1], limit=limit)
    
    print("PROCESS UP ...")
    upX, upY = SingleFolderProcess(path + "/UP/", [1, 0], limit=limit)

    X = np.array(downX + upX)
    Y = np.array(downY + upY)

    return X, Y

def GetDataGenerator(path, batch_size):
    imageGenerator = ImageDataGenerator(rescale=1./255)

    train_generator = imageGenerator.flow_from_directory(
        path,
        target_size=(224,224),
        batch_size=batch_size,
        class_mode='categorical'
    )

    return train_generator

if __name__ == "__main__":
    dataset = DataProcess(os.getcwd() + "/btc-trading-patterns/train")

    print(dataset[1][0])
    plt.imshow(dataset[0][0])
    plt.show()