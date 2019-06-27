import keras
import math
import os, sys, getopt
from keras.models import load_model
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout, Conv2D, MaxPooling2D, GlobalMaxPooling2D
from keras.preprocessing import image
from keras.callbacks import ModelCheckpoint
from keras.backend.tensorflow_backend import set_session
from optimizers.AdamW import AdamW

batch_size = 32

def get_generator():
    dataset_gen = image.ImageDataGenerator(
        rescale = 1./255,
        shear_range = 0.2,
        zoom_range = 0.2,
        width_shift_range = 0.1,
        height_shift_range = 0.1,
        horizontal_flip = True,
        rotation_range = 40,
        validation_split = 0.2
    )

    dataset_genrator = dataset_gen.flow_from_directory(
        'dataset',
        target_size = (256, 256),
        color_mode = 'rgb',
        batch_size = batch_size,
        class_mode = 'categorical',
        interpolation='lanczos',
        subset='training'
    )

    validation_generator = dataset_gen.flow_from_directory(
        'dataset',
        target_size = (256, 256),
        color_mode = 'rgb',
        batch_size = batch_size,
        class_mode = 'categorical',
        interpolation='lanczos',
        subset='validation'
    )
    return dataset_genrator, validation_generator

if __name__ == "__main__":
    # get model and generators
    model_file = ""
    run_name = ""
    epoch = 1000
    steps = 10
    validation_steps = 2

    # parse args
    try:
        opts, args = getopt.getopt(sys.argv[1:], "hm:s:",["model=","name=","epoch=","steps=","validation-steps="])
    except getopt.GetoptError:
        print("python evaluate.py -m <model.hdf5> -s <steps>")
    for opt, arg in opts:
        if opt == '-h':
            print("python evaluate.py -m <model.hdf5> -s <steps>")
            sys.exit()
        elif opt in ('-m', '--model'):
            model_file = arg
        elif opt in ('-s', '--steps'):
            steps = int(arg)

    model = load_model(model_file)
    training_genrator, validation_generator = get_generator()

    # evaluate
    loss, accuracy = model.evaluate_generator(validation_generator, steps=steps)
    print('[INFO] test loss: %.6f, acc: %.6f'%(loss,accuracy))