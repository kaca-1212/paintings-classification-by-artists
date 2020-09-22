import h5py
import numpy as np
from matplotlib import pyplot
from sklearn.model_selection import GridSearchCV
from joblib import dump, load
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.svm import SVC
from sklearn.decomposition import PCA
import time

features_test = ['gist', 'haralick', 'hist', 'hog', 'hu', 'SIFT']
enable = [1, 0, 1, 0, 1, 0]

label_file = h5py.File('./output/svm/hu_labels.h5', 'r')
labels = np.array(label_file['dataset_1'])

features = np.array([])

# Load the features specified by enable
for i, feature in enumerate(features_test):
    if not enable[i]:
        continue
    feature_file = h5py.File('./output/svm/' + feature + '_features.h5', 'r')

    if features.size == 0:
        features = np.array(feature_file['dataset_1'])
    else:
        features = np.hstack([features, np.array(feature_file['dataset_1'])])

labels = labels.flatten()
print(features.shape)
feature_count = min(features.shape[1], 100)
pca = PCA(n_components=feature_count)

features = pca.fit_transform(features)
val_splitter = StratifiedShuffleSplit(1, 0.11, 0.88)
train_indices, val_indices = next(val_splitter.split(features, labels))

(train_data, train_label, val_data, val_label) = features[train_indices], labels[train_indices], features[val_indices], \
                                                 labels[val_indices]

# Fit the PCA on the training data, then additionally transform the validation data
train_data = pca.fit_transform(train_data)
val_data = pca.transform(val_data)

# Form the parameter grid for the grid search
param_grid = [
   {'C': [.00001, .0001, .001, .01, .1, 1, 10], 'gamma':[.00001, .0001, .001, .01, .1, 1, 10], 'kernel':['linear']}
]

# The SVM after params are found from grid search
# model = SVC(C=10, gamma=.1, kernel='linear', degree=4)
model = GridSearchCV(SVC(), param_grid)

# Time the training
start_time = time.time()
model.fit(train_data, train_label)
train_time = time.time() - start_time
print("Train time: " + str(train_time))

dump(model,filename='./output/svm/svm.model')


# Time the inference
start_time = time.time()
y_pred = model.predict(val_data)
inf_time = time.time() - start_time

# Show the confusion matrix
print(confusion_matrix(val_label, y_pred))

print("Inf time: " + str(inf_time))
print("Accuracy:", model.score(val_data, val_label))

print(classification_report(val_label, y_pred))