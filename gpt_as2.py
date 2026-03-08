"""
OMR Bubble Detection using Machine Learning
--------------------------------------------

Input image size: 47 x 365

Pipeline:
1. Detect bubble region
2. Crop question number
3. Split 4 bubbles
4. Resize bubble
5. ML classify filled / empty
"""

import cv2
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


# ===============================
# STEP 1: PREPROCESS IMAGE
# ===============================

def preprocess(image):

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # blur = cv2.GaussianBlur(gray,(5,5),0)

    thresh = cv2.threshold(
        gray,
        150,
        255,
        cv2.THRESH_BINARY_INV
    )[1]

    return thresh


# ===============================
# STEP 2: FIND BUBBLE START
# ===============================

def detect_bubble_region(thresh):

    # tính tổng pixel đen theo cột
    col_sum = np.sum(thresh, axis=0)

    # tìm cột đầu tiên có nhiều pixel đen
    bubble_start = np.argmax(col_sum > 500)

    return bubble_start


# ===============================
# STEP 3: SPLIT 4 BUBBLES
# ===============================

def split_bubbles(thresh):

    h,w = thresh.shape

    bubble_width = w // 4

    bubbles = []

    for i in range(4):

        x1 = i * bubble_width
        x2 = (i+1) * bubble_width

        bubble = thresh[:,x1:x2]

        bubbles.append(bubble)

    return bubbles


# ===============================
# STEP 4: CONVERT TO FEATURE
# ===============================

def bubble_to_feature(bubble):

    # resize giảm feature
    bubble = cv2.resize(bubble,(16,16))

    # flatten
    feature = bubble.flatten()

    return feature


# ===============================
# STEP 5: LOAD DATASET
# ===============================

def load_dataset(dataset_path):

    X = []
    y = []

    import os

    labels = ['filled','empty']

    for label in labels:

        folder = dataset_path + "/" + label

        for file in os.listdir(folder):

            path = folder + "/" + file

            img = cv2.imread(path)

            thresh = preprocess(img)

            feature = bubble_to_feature(thresh)

            X.append(feature)

            y.append(label)

    return np.array(X), np.array(y)


# ===============================
# STEP 6: TRAIN MODEL
# ===============================

def train_model(X,y):

    X_train,X_test,y_train,y_test = train_test_split(
        X,y,
        test_size=0.2
    )

    model = LogisticRegression(max_iter=1000)

    model.fit(X_train,y_train)

    print("Accuracy:",model.score(X_test,y_test))

    return model


# ===============================
# STEP 7: DETECT ANSWER
# ===============================

def detect_answer(image_path,model):

    image = cv2.imread(image_path)

    image = cv2.resize(image,(365,47))

    thresh = preprocess(image)

    bubble_start = detect_bubble_region(thresh)

    # crop bỏ phần số
    bubble_img = thresh[:,bubble_start:]

    bubbles = split_bubbles(bubble_img)

    predictions = []

    for bubble in bubbles:

        feature = bubble_to_feature(bubble)

        pred = model.predict([feature])[0]

        predictions.append(pred)

    answers = ['A','B','C','D']

    for i,p in enumerate(predictions):

        if p == 'filled':
            return answers[i]

    return None


# ===============================
# MAIN
# ===============================

if __name__ == "__main__":

    # load dataset
    X,y = load_dataset("dataset")

    # train model
    model = train_model(X,y)

    # test image
    ans = detect_answer("test.png",model)

    print("Detected answer:",ans)