import os
import cv2
import numpy as np
import pickle

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

dataset_path = r"D:\Coding\MSE35HN\MLE501\data_cleaned"

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

            # resize
            img = cv2.resize(img, (resize_width, resize_height))

            # normalize
            img = img / 255.0

            # feature
            feature_vector = img.flatten()
            # hoặc thử:
            # feature_vector = np.mean(img, axis=0)

            X.append(feature_vector)
            y.append(label)

X = np.array(X)
y = np.array(y)

print("Dataset shape:", X.shape)
print("Labels:", np.unique(y))


# chia train/test
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# ===== RANDOM FOREST =====
model = RandomForestClassifier(
    n_estimators=200,     # số cây
    max_depth=None,       # không giới hạn depth
    random_state=42,
    n_jobs=-1             # dùng toàn bộ CPU
)

model.fit(X_train, y_train)


# predict
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)


# save model
with open("../omr_model_random_fs.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model saved: omr_model_random_fs.pkl")


# evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))

print(classification_report(
    y_test,
    y_pred,
    target_names=["A","B","C","D","Blank"]
))

print("Train accuracy:", model.score(X_train, y_train))
print("Test accuracy:", model.score(X_test, y_test))