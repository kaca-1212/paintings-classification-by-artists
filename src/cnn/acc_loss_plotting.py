from matplotlib import pyplot as plt
import os
import json

dir_path ='./output/cnn/histories/'

allModels = os.listdir(dir_path)
data = {}

for file in allModels:
    with open(dir_path + file) as json_file:
        data.update({file: json.load(json_file)})

for model in data:
    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.title('Loss')
    plt.plot(data.get(model)['loss'].keys(), data.get(model)['loss'].values(), label='train')
    plt.plot(data.get(model)['val_loss'].keys(), data.get(model)['val_loss'].values(), label='val')
    plt.legend(loc='best')

    plt.subplot(1, 2, 2)
    plt.title('Accuracy')
    plt.plot(data.get(model)['accuracy'].keys(), data.get(model)['accuracy'].values(), label='train')
    plt.plot(data.get(model)['val_accuracy'].keys(), data.get(model)['val_accuracy'].values(), label='val')
    plt.legend(loc='best')
    plt.show()
