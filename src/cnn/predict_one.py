
import tensorflow as tf
import argparse
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.image as mpimg


batch_size = 32
fpath = './dataset/cnn_dataset/test/Albrecht Durer/Albrecht Durer_82.jpg'
artists_list = ["Albrecht Durer",
                "Boris Kustodiev",
                "Camille Corot",
                "Camille Pissarro",
                "Childe Hassam",
                "Claude Monet",
                "Edgar Degas",
                "Eugene Boudin",
                "Giovanni Battista Piranesi",
                "Gustave Dore",
                "Henri Matisse",
                "Ilya Repin",
                "Ivan Aivazovsky",
                "Ivan Shishkin",
                "John Singer Sargent"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(dest='neural_net_type', choices=['densenet121', 'resnet50v2'],
                        help="Which type of neural network you want to load?")
    parser.add_argument(dest='model', help='Path for weights file.')

    # Parse and print the results
    args = parser.parse_args()
    model = tf.keras.models.load_model(args.model)

    if args.neural_net_type.lower() == 'densenet121':
        prep_func = tf.keras.applications.densenet.preprocess_input
        shape = (224,224)
    elif args.neural_net_type.lower() == 'resnet50v2':
        prep_func = tf.keras.applications.resnet_v2.preprocess_input
        shape = (299,299)
    else:
        print('No valid neural network type entered! Exiting.')
        return -1

    # preprocessing input
    img = tf.keras.preprocessing.image.load_img(fpath, target_size=shape)
    y = tf.keras.preprocessing.image.img_to_array(img)
    y = np.expand_dims(y, axis=0)
    y = prep_func(y)
    predicted = np.argmax(model.predict(y), axis=-1)
    print(artists_list[predicted[0]])

    img1 = mpimg.imread(fpath)
    imgplot = plt.imshow(img1)
    plt.show()

main()