from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
import pandas as pd
from tensorflow.keras.utils import multi_gpu_model
import tensorflow as tf


# File used for training neural networks. Some changes are parametrized, but model was changed
# manually.

batch_size = 32
version_suffix = '_50v2'
output_dir = 'output/cnn/'

preprocessing_func = tf.keras.applications.resnet_v2.preprocess_input
trained_model = tf.keras.applications.ResNet152V2(
  classes=1000, weights='imagenet', include_top=False, input_shape=(299,299,3)
)

def make_fit_gen():
    train_datagen = ImageDataGenerator(
#        rotation_range=180,
#        zoom_range=0.2,
#        width_shift_range=0.2,
#        height_shift_range=0.2,
#        shear_range=0.3,
#        horizontal_flip=True,
#        vertical_flip=True,
#        fill_mode='reflect',
	preprocessing_function=preprocessing_func)

    val_datagen = ImageDataGenerator(preprocessing_function=preprocessing_func)

    train_generator = train_datagen.flow_from_directory('dataset/cnn_dataset/train', target_size=(299,299),
                                                        batch_size=batch_size, class_mode='categorical')
    val_generator = val_datagen.flow_from_directory('dataset/cnn_dataset/validation', target_size=(299,299),
                                                    batch_size=batch_size, class_mode='categorical')

    return (train_generator, val_generator)

def compile_model(model):
    adam = Adam(lr=0.0002)
    regularizer = tf.keras.regularizers.l2(0.0015)
    # lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    #     initial_learning_rate=1e-2,
    #     decay_steps=10000,
    #     decay_rate=0.9)

    # adam=Adam(lr=lr_schedule)
    for layer in model.layers:
        for attr in ['kernel_regularizer']:
            if hasattr(layer, attr):
                setattr(layer, attr, regularizer)
    model.compile(
        loss='categorical_crossentropy',
        optimizer=adam,
        metrics=['accuracy'])
    return model

model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath='{output_dir}best{version_suffix}.h5'.format(output_dir=output_dir,version_suffix=version_suffix),
    save_weights_only=True,
    monitor='val_accuracy',
    mode='max',
    save_best_only=True)

early_stop = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=6)

model = tf.keras.Sequential()
model.add(trained_model)
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(15, activation='softmax'))

model = compile_model(model)
gens = make_fit_gen()
history = model.fit(gens[0], epochs=200, validation_data=gens[1], callbacks=[model_checkpoint_callback, early_stop])

# convert the history.history dict to a pandas DataFrame:
hist_df = pd.DataFrame(history.history)

# save to json:
hist_json_file = '{output_dir}history_{version_suffix}.json'.format(version_suffix=version_suffix, output_dir=output_dir)

with open(hist_json_file, mode='w') as f:
    hist_df.to_json(f)
model.save_weights('{output_dir}first_{version_suffix}.h5'.format(version_suffix=version_suffix, output_dir=output_dir))
