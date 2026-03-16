import os
import cv2
import numpy as np
import pickle
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.multiclass import OneVsRestClassifier

dataset_path = r"D:\Coding\MSE35HN\MLE501\data_cleaned_2"

X = []
y = []

resize_width = 60
resize_height = 15

labels = ["A", "B", "C", "D", "Blank"]

for label in labels:

    folder = os.path.join(dataset_path, label)

    for filename in os.listdir(folder):

        if filename.endswith(".png"):
            file_path = os.path.join(folder, filename)

            # đọc ảnh grayscale
            img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)

            # resize ảnh
            img = cv2.resize(img, (resize_width, resize_height))

            # normalize về 0-1
            img = img / 255.0

            # flatten 2D -> 1D
            feature_vector = img.flatten()

            X.append(feature_vector)

            one_hot = [0, 0, 0, 0, 0]
            one_hot[labels.index(label)] = 1

            y.append(one_hot)

X = np.array(X)
y = np.array(y)

print("Dataset shape:", X.shape)
print("Labels:", y.shape)

# chia train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=np.argmax(y, axis=1)
)

# train model
model = OneVsRestClassifier(
    SVC(kernel="linear", probability=True)
)

model.fit(X_train, y_train)

# predict
y_pred = model.predict(X_test)

# save model
with open("omr_model_svm.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model saved: omr_model_svm.pkl")

print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred, target_names=["A","B","C","D","Blank"]))
print("Train accuracy:", model.score(X_train, y_train))
print("Test accuracy:", model.score(X_test, y_test))
