from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import argparse
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np

batch_size = 32

def compile_model(model):
    adam = Adam(lr=0.0002)
    model.compile(
        loss='categorical_crossentropy',
        optimizer=adam,
        metrics=['accuracy'])
    return model


def make_test_gen(preprocess_func):
    test_datagen = ImageDataGenerator(preprocessing_function=preprocess_func)

    test_generator = test_datagen.flow_from_directory('dataset/cnn_dataset/test', target_size=(224, 224),
                                                      batch_size=batch_size, class_mode='categorical', shuffle=False)
    return test_generator


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(dest='neural_net_type', choices=['densenet121', 'resnet50v2'],
                        help="Which type of neural network you want to load?")
    parser.add_argument(dest='weights_file', help='Path for weights file.')

    # Parse and print the results
    args = parser.parse_args()

    if args.neural_net_type.lower() == 'densenet121':
        test_gen = make_test_gen(tf.keras.applications.densenet.preprocess_input)
        trained_model = tf.keras.applications.DenseNet121(
            classes=1000, weights='imagenet', include_top=False, input_shape=(224, 224, 3)
        )
    elif args.neural_net_type.lower() == 'resnet50v2':
        test_gen = make_test_gen(tf.keras.applications.resnet_v2.preprocess_input)
        trained_model = tf.keras.applications.ResNet50V2(
            classes=1000, weights='imagenet', include_top=False, input_shape=(299, 299, 3)
        )
    else:
        print('No valid neural network type entered! Exiting.')
        return -1

    model = tf.keras.Sequential()
    model.add(trained_model)
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(15, activation='softmax'))
    model = compile_model(model)

    model.load_weights(args.weights_file)

    score = model.evaluate(test_gen)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    probabilities = model.predict(test_gen)
    y_true = test_gen.classes
    y_predicted_classes = np.argmax(probabilities, axis=1)

    cm = confusion_matrix(y_true, y_predicted_classes)
    print(cm)
    report = classification_report(y_true, y_predicted_classes)
    print(report)

    model.save('output/cnn/{}.model'.format(args.neural_net_type.lower()))

main()