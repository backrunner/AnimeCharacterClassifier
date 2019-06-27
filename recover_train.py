import keras
import sys, getopt, os
from keras.preprocessing import image
from keras.models import load_model
from keras.callbacks import ModelCheckpoint
from optimizers.AdamW import AdamW

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
        batch_size = 32,
        class_mode = 'categorical',
        interpolation='lanczos',
        subset='training'
    )

    validation_generator = dataset_gen.flow_from_directory(
        'dataset',
        target_size = (256, 256),
        color_mode = 'rgb',
        batch_size = 32,
        class_mode = 'categorical',
        interpolation='lanczos',
        subset='validation'
    )
    return dataset_genrator, validation_generator

def get_tensorboard_callback(run):
    tensorboard = keras.callbacks.TensorBoard(log_dir='./logs/'+run,
        batch_size=32,
        update_freq=500,
        write_graph=True,
        write_images=True)
    return tensorboard

if __name__ == "__main__":
    model_file = ""
    run_name = ""
    epoch = 1000
    steps = 10
    validation_steps = 2

    # parse args
    try:
        opts, args = getopt.getopt(sys.argv[1:], "hm:n:e:s:v:",["model=","name=","epoch=","steps=","validation-steps="])
    except getopt.GetoptError:
        print("python recover_train.py -m <model.hdf5> -n <run_name> -e <epoch> -s <steps>")
    for opt, arg in opts:
        if opt == '-h':
            print("python recover_train.py -m <model.hdf5> -n <run_name> -e <epoch> -s <steps>")
            sys.exit()
        elif opt in ('-m', '--model'):
            model_file = arg
        elif opt in ('-n', '--name'):
            run_name = arg
        elif opt in ('-e', '--epoch'):
            epoch = int(arg)
        elif opt in ('-s', '--steps'):
            steps = int(arg)
        elif opt in ('-v', '--validation-steps'):
            validation_steps = int(arg)

    model = load_model(model_file)

    # reset optimizer and recompile model
    optimizer = keras.optimizers.SGD(lr=1e-6, momentum=0., decay=0., nesterov=False)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer,metrics=['accuracy'])

    # get generator
    train_generator, validation_generator = get_generator()

    checkpoint = ModelCheckpoint("models/checkpoint-{epoch:04d}e-val_acc_{val_acc:.5f}.hdf5", monitor='val_acc', verbose=1, 
        save_best_only=True)

    # use tensorboard
    tensorboard = get_tensorboard_callback(run_name)
    callbacks = [checkpoint, tensorboard]

    # train model
    model.fit_generator(
        train_generator,
        steps_per_epoch=steps,
        epochs=epoch,
        callbacks=callbacks,
        validation_data=validation_generator,
        validation_steps=validation_steps)

    # evaluate model
    loss, accuracy = model.evaluate_generator(validation_generator, steps=100)
    print('[INFO] Train finished, test loss: %.6f, acc: %.6f'%(loss,accuracy))

    model.save('models/model_recover_train_'+ run_name +'_final.hdf5')