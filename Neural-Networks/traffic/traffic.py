import cv2
import numpy as np
import os
import sys
import tensorflow as tf

from sklearn.model_selection import train_test_split

EPOCHS = 10
IMG_WIDTH = 30
IMG_HEIGHT = 30
NUM_CATEGORIES = 43
TEST_SIZE = 0.4


def main():

    # Check command-line arguments
    if len(sys.argv) not in [2, 3]:
        sys.exit("Usage: python traffic.py data_directory [model.h5]")

    # Get image arrays and labels for all image files
    images, labels = load_data(sys.argv[1])

    # Split data into training and testing sets
    labels = tf.keras.utils.to_categorical(labels)
    x_train, x_test, y_train, y_test = train_test_split(
        np.array(images), np.array(labels), test_size=TEST_SIZE
    )

    # Get a compiled neural network
    model = get_model()

    # Fit model on training data
    model.fit(x_train, y_train, epochs=EPOCHS)

    # Evaluate neural network performance
    model.evaluate(x_test,  y_test, verbose=2)

    # Save model to file
    if len(sys.argv) == 3:
        filename = sys.argv[2]
        model.save(filename)
        print(f"Model saved to {filename}.")


def load_data(data_dir):
    """
    Load image data from directory `data_dir`.

    Assume `data_dir` has one directory named after each category, numbered
    0 through NUM_CATEGORIES - 1. Inside each category directory will be some
    number of image files.

    Return tuple `(images, labels)`. `images` should be a list of all
    of the images in the data directory, where each image is formatted as a
    numpy ndarray with dimensions IMG_WIDTH x IMG_HEIGHT x 3. `labels` should
    be a list of integer labels, representing the categories for each of the
    corresponding `images`.
    """

    sub_dirs = os.listdir(data_dir)

    images = []
    labels = []

    for dir in sub_dirs[1:]:

        new_path = os.path.join(data_dir, dir)
        all_images = os.listdir(new_path)    # ensures that files are in ascending order based on name


        for img in all_images:
            current_img_path = os.path.join(data_dir, dir , img)
            current_img = cv2.imread(current_img_path)
            processed_img = cv2.resize(current_img, (30, 30))
            images.append(processed_img)
            labels.append(int(dir))  # keep adding corresponding labels

    return (images, labels)



def get_model():
    """
    Returns a compiled convolutional neural network model. Assume that the
    `input_shape` of the first layer is `(IMG_WIDTH, IMG_HEIGHT, 3)`.
    The output layer should have `NUM_CATEGORIES` units, one for each category.
    """

    '''    
     x is the images and y is the label
    
    (x_train, y_train), (x_text, y_test) = load_data()
    # convert pixel range into 0 and 1 (remember 3b1b's video?) 
    x_train, x_test = x_train / 255.0 , x_test/255.0

    # One hot encoding on the categories
    y_train = tf.keras.utils.to_categorical(y_train)
    y_test = tf.keras.utils.to_categorical(y_test)
    
    
    x_train = x_train.reshape(
        x_train.shape[0], x_train.shape[1], x_train.shape[2], 3
    )

    x_test = x_test.reshape(
        x_test.shape[0], x_test.shape[1], x_test.shape[2], 3
    )
    '''
    
    model = tf.keras.models.Sequential([

        # Convolution layer
        tf.keras.layers.Conv2D(
            16, (3,3), activation="relu", input_shape=(30, 30, 3)
        ),

        # pooling
        tf.keras.layers.MaxPooling2D(pool_size=(2,2)),

        # Flatten units
        tf.keras.layers.Flatten(),
        
        tf.keras.layers.Dense(4, activation="sigmoid"),
        tf.keras.layers.Dropout(0.5),

        # output layer
        tf.keras.layers.Dense(3, activation="softmax")

    ])

    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model

if __name__ == "__main__":
    main()
