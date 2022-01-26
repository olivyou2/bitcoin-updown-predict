import tensorboard
from data_processing import DataProcess, GetDataGenerator
from model import GetModel
from tensorflow.python.keras.callbacks import TensorBoard
from keras.callbacks import ModelCheckpoint, EarlyStopping
from time import time
import os

from sys import getsizeof

if __name__ == "__main__":
    path = os.getcwd()

    trainGenerator = GetDataGenerator(path + "/patterns/train", batch_size=1)
    testGenerator = GetDataGenerator(path + "/patterns/test", batch_size=1)

    batch_size=32
    batches = 8
    epoch = 16

    model = GetModel()

    tensorboardcall = TensorBoard(log_dir="{}/logs/{}".format(path, time()))
    checkpoint = ModelCheckpoint("btc.h5", monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)
    early = EarlyStopping(monitor='acc', min_delta=0, patience=20, verbose=1, mode='auto')
    
    #for i in range (batches):
    #print("BATCH", i, "/", batches)
    trainX, trainY = trainGenerator.next()
    print(trainX.nbytes)

    #test = testGenerator.next()

    hist = model.fit(
        batch_size=1,
        x=trainX,
        y=trainY,
        epochs=epoch,
        #validation_data=test,
        callbacks = [checkpoint,early,tensorboardcall],
        shuffle=True)