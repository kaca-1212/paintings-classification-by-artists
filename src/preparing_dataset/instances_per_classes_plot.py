import os
import matplotlib.pyplot as plt; plt.rcdefaults()
import numpy as np
import matplotlib.pyplot as plt

data_path = './svm_set'
artist_dict = {"Albrecht Durer": 0,
                "Boris Kustodiev": 0,
                "Camille Corot": 0,
                "Camille Pissarro": 0,
                "Childe Hassam": 0,
                "Claude Monet": 0,
                "Edgar Degas": 0,
                "Eugene Boudin": 0,
                "Giovanni Battista Piranesi": 0,
                "Gustave Dore": 0,
                "Henri Matisse": 0,
                "Ilya Repin": 0,
                "Ivan Aivazovsky": 0,
                "Ivan Shishkin": 0,
                "John Singer Sargent": 0
}

for ind, f in enumerate(os.listdir(data_path)):
    temp = f.split("_")
    artist_dict[temp[0]] = artist_dict[temp[0]]+1


objects = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
y_pos = np.arange(len(objects))
performance = artist_dict.values()

plt.bar(y_pos, performance,align='center', alpha=0.5)
plt.xticks(y_pos, objects)
plt.ylabel('# Samples')
plt.title('Artist')

plt.show()



print(artist_dict)