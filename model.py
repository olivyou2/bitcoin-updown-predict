from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.utils import plot_model
from keras import models, layers, regularizers, optimizers

def GetModel():
    model = models.Sequential()
    model.add(VGG16(weights="imagenet"))

    model.add(layers.Flatten())
    model.add(layers.Dense(500, 
        kernel_regularizer=regularizers.l1_l2(l1=0.001, l2=0.001), activation="relu"))
    model.add(layers.Dropout(0.5))

    model.add(layers.Dense(300, 
        kernel_regularizer=regularizers.l1_l2(l1=0.001, l2=0.001), activation="relu"))
    model.add(layers.Dropout(0.5))

    model.add(layers.Dense(100, 
        kernel_regularizer=regularizers.l1_l2(l1=0.001, l2=0.001), activation="relu"))
    model.add(layers.Dropout(0.5))

    model.add(layers.Dense(2, activation="softmax"))
    model.compile(loss="binary_crossentropy", optimizer=optimizers.rmsprop_v2.RMSprop(lr=1e-6), metrics=["acc"])

    print(model.summary())

    return model