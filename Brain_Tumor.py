import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.svm import SVC 
import pickle

# Prepare and Collect Data
path = os.listdir("c:\\Users\\FORMAT\\Desktop\\Project\\data\\Training\\")
classes = {'no_tumor':0,'pituitary_tumor':1}

X = [] # data
Y = [] # target

for cls in classes:
    pth = 'c:\\Users\\FORMAT\\Desktop\\Project\\data\\Training\\' + cls
    for j in os.listdir(pth):
        img = cv2.imread(pth+'\\'+j, 0) # read and convert image to grayscale
        img = cv2.resize(img, (64, 64)) # resize image to 64 x 64
        X.append(img)
        Y.append(classes[cls])

z = np.unique(Y) # return classes
print("Classes ", z)
print("-------------------")

X = np.array(X)
Y = np.array(Y)
v = pd.Series(Y).value_counts() # return number of elements that belong to each class
print("Number of elements that belong to each class")
print(v)
print("-------------------")

print("X shape")
print(X.shape)
print("-------------------")

X_updated = X.reshape(len(X),-1)
print("X shape")
print(X_updated.shape)
print("-------------------")

# Split
x_train, x_test, y_train, y_test = train_test_split(X_updated, Y, test_size=0.2, random_state=10)
print("After spliting data")
print("x train ", x_train.shape)
print("x test ", x_test.shape)
print("y train ", y_train.shape)
print("y test ", y_test.shape)
print("-------------------")

# Feature Scaling
print("Feature Scaling")
print(x_train.max(), x_train.min())
print(x_test.max(), x_test.min())
x_train = x_train/255
x_test = x_test/255
print(x_train.max(), x_train.min())
print(x_test.max(), x_test.min())

# Train Model
sv = SVC()
sv.fit(x_train,y_train)

with open('brain_model','wb') as f:
    pickle.dump(sv,f)

# Prediction
pred = sv.predict(x_test)
print("Predicted Values")
print(pred)
print("Actual Values")
print(y_test)
not_accure = np.where(y_test!=pred)
print("Predicted Values are not true")
print(not_accure)
print("-------------------")

# Evaluation
print("Evaluation")
print("Trainig Score: ",sv.score(x_train,y_train))
print("Testing Score: ",sv.score(x_test,y_test))
print("Accuracy", accuracy_score(y_test, pred))
print("Confusion Matrix\n", confusion_matrix(y_test, pred))
print("-------------------")
ConfusionMatrixDisplay(confusion_matrix= confusion_matrix(y_test, pred),display_labels = sv.classes_).plot()

# Test Model
dec = {0:'No Tumor', 1:'Positive Tumor'}
plt.figure(figsize=(12,8))
c = 1
for i in os.listdir("c:\\Users\\FORMAT\\Desktop\\Project\\data\\Testing\\no_tumor\\")[:9]:
    plt.subplot(3,3,c)
    img = cv2.imread("c:\\Users\\FORMAT\\Desktop\\Project\\data\\Testing\\no_tumor\\" + i,0)
    img1 = cv2.resize(img,(64,64))
    img2 = img1.reshape(1,-1)/255
    p = sv.predict(img2)
    print(p)
    plt.title(dec[p[0]])
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)) 
    plt.axis('off')
    c+=1
plt.show()

print("-------------------")
plt.figure(figsize=(12,8))
cc = 1
for i in os.listdir("c:\\Users\\FORMAT\\Desktop\\Project\\data\\Testing\\pituitary_tumor\\")[0:9]:
    plt.subplot(3,3,cc)
    img = cv2.imread("c:\\Users\\FORMAT\\Desktop\\Project\\data\\Testing\\pituitary_tumor\\" + i,0)
    img1 = cv2.resize(img,(64,64))
    img2 = img1.reshape(1,-1)/255
    p = sv.predict(img2)
    print(p)
    plt.title(dec[p[0]])
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)) 
    plt.axis('off')
    cc+=1
plt.show()
