from joblib import load
import gist
import cv2
import numpy as np

model = load('./output/svm/svm.model')
fpath = './dataset/cnn_dataset/test/Albrecht Durer/Albrecht Durer_7.jpg'
fixed_size = tuple((256, 256))

def fd_gist(image):
    print("gist")
    return gist.extract(image)

def fd_histogram(image, mask=None):
    print("hist")
    # convert the image to HSV color-space
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # compute the color histogram
    hist = cv2.calcHist([image], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    # normalize the histogram
    cv2.normalize(hist, hist)
    # return the histogram
    return hist.flatten()

def fd_hu_moments(image):
    print("hu")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    feature = cv2.HuMoments(cv2.moments(image)).flatten()
    return feature

img = cv2.imread(fpath)

features = np.array([])

f1 = fd_gist(img)
features = np.hstack([features, f1])

f1 = fd_histogram(img)
features = np.hstack([features, f1])

f1 = fd_hu_moments(img)
features = np.hstack([features, f1])

res = model.predict([1,features])

print (res)