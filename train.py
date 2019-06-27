import keras
import math
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

def get_model(classes_num, input_shape):
    model = Sequential()

    model.add(Conv2D(32,(3, 3), strides=(1,1), activation="relu",
        input_shape=input_shape, padding='same'))
    model.add(Conv2D(32,(3, 3), strides=(1,1), activation="relu", padding='same'))
    model.add(Conv2D(48,(3, 3), strides=(1,1), activation="relu", padding='same'))
    model.add(Conv2D(48,(3, 3), strides=(1,1), activation="relu", padding='same'))
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64,(3, 3), strides=(1,1), activation="relu", padding='same'))
    model.add(Conv2D(64,(3, 3), strides=(1,1), activation="relu", padding='same'))
    model.add(Conv2D(64,(3, 3), strides=(1,1), activation="relu", padding='same'))
    model.add(Conv2D(64,(3, 3), strides=(1,1), activation="relu", padding='same'))
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(80,(3, 3), strides=(1,1), activation="relu", padding='same'))
    model.add(Conv2D(80,(3, 3), strides=(1,1), activation="relu", padding='same'))
    model.add(Conv2D(128,(3, 3), strides=(1,1), activation="relu", padding='same'))
    model.add(Conv2D(128,(3, 3), strides=(1,1), activation="relu", padding='same'))
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(256, activation="relu"))
    model.add(Dropout(0.25))
    model.add(Dense(classes_num, activation="softmax"))

    print('[INFO] Compiling the model...')
    optimizer = AdamW(lr=3e-4, beta_1=0.9, beta_2=0.999, epsilon=None, decay=2e-8, weight_decay=0.025, batch_size=32, samples_per_epoch=384, epochs=500)

    model.compile(loss='categorical_crossentropy', optimizer=optimizer,metrics=['accuracy'])

    return model

def get_tensorboard_callback(run):
    tensorboard = keras.callbacks.TensorBoard(log_dir='./logs/'+run,
        batch_size=batch_size,
        update_freq=500,
        write_graph=True,
        write_images=True)
    return tensorboard

if __name__ == "__main__":
    # get model and generators
    model = get_model(3, (256, 256, 3))
    training_genrator, validation_generator = get_generator()

    # save checkpoint
    checkpoint = ModelCheckpoint("models/checkpoint-{epoch:04d}e-val_acc_{val_acc:.5f}.hdf5", monitor='val_acc', verbose=1, 
        save_best_only=True)

    # use tensorboard
    tensorboard = get_tensorboard_callback("adamw_3e-4_newsettings")
    callbacks = [checkpoint, tensorboard]

    # train model
    model.fit_generator(
        training_genrator,
        steps_per_epoch=12,
        epochs=500,
        callbacks=callbacks,
        validation_data=validation_generator,
        validation_steps=3)

    # evaluate
    loss, accuracy = model.evaluate_generator(validation_generator, steps=100)
    print('[INFO] Train finished, test loss: %.6f, acc: %.6f'%(loss,accuracy))

    # save final model
    model.save('models/model_train_finished.hdf5')