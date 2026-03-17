import os

import cv2
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

best_n = 0
best_acc = 0

results = []

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

for n in range(10, 301, 10):  # từ 10 đến 300, bước 10

    model = RandomForestClassifier(
        n_estimators=n,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    results.append((n, acc))

    print(f"n_estimators={n} → accuracy={acc:.4f}")

    if acc > best_acc:
        best_acc = acc
        best_n = n

print("\nBest n_estimators:", best_n)
print("Best accuracy:", best_acc)