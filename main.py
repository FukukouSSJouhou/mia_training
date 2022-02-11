
import os
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint
#from tensorflow.keras.utils import np_utils
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
import numpy as np
from sklearn.model_selection import train_test_split
from PIL import Image
import glob
from tensorflow.keras.preprocessing import image
def mainmakemodel5ex():
    folder_path = './train/'
    folder = ["angry", "happy", "neutral", "sad", "surprise"]
    n_class = 5  # the number of folders in 'folder'
    image_size = 48

    # To read dataset
    X = []
    Y = []
    for index, name in enumerate(folder):
        dir = folder_path + name
        files = glob.glob(dir + "/*.jpg")
        for i, file in enumerate(files):
            #        img = image.load_img(file, grayscale=True , target_size=(image_size, image_size))
            #        img = image.load_img(file, grayscale=False , target_size=(image_size, image_size))
            print("Image loading {0}-No.{1}".format(name, i + 1))
            img = image.load_img(file, grayscale=False, color_mode="grayscale", target_size=(image_size, image_size))
            data = image.img_to_array(img)
            X.append(data)
            Y.append(index)

    X = np.array(X)
    Y = np.array(Y)

    X = X.astype('float32')
    X = X / 255.0
    Y = tf.keras.utils.to_categorical(Y, n_class)

    # Divide the dataset into test data and training data
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.20)
    # X_train, X_varid, y_train, y_varid = train_test_split(X, Y, test_size=0.20)

    print("This is ", X_train.shape[1:])
    # CNN layers definition
    model = Sequential()

    model.add(Conv2D(32, (3, 3), padding='same', input_shape=X_train.shape[1:],name="1stslotS"))
    model.add(Activation('relu'))
    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(n_class))
    model.add(Activation('softmax',name='output34'))
    #model.summary()
    model.compile(loss='categorical_crossentropy', optimizer='SGD', metrics=['accuracy'])

    #tf.keras.utils.plot_model(model, show_shapes=True, expand_nested=True,to_file='model.png',dpi=96)
    return  1
    MODEL_DIR = './models/every_epoch_models'
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
    # checkpoint = ModelCheckpoint(filepath=os.path.join(MODEL_DIR, "model-{epoch:02d}.h5"), save_best_only=True)  # 精度が向上した場合のみ保存する。
    checkpoint = ModelCheckpoint(filepath=os.path.join(MODEL_DIR, "model-stage"),
                                 save_best_only=False)  # 精度の向上・未向上に関わらず保存する。
    tensorboardkun=tf.keras.callbacks.TensorBoard(log_dir='./logs', histogram_freq=0, write_graph=True)
    # Start training
    # epochs = 100
    epochs = 300
    batch_size = 32
    history = model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1,
                        callbacks=[checkpoint,tensorboardkun])

    # グラフ描画

    # Save the tained model
    model.save("./models/5face_emotions_~~ep_full")

    # The final evaluation of the traind model as a hdf5 file
    score = model.evaluate(X_test, y_test)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
def main():
    print("Tdn")
if __name__ == '__main__':
    main()