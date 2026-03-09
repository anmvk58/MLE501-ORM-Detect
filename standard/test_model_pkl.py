import cv2
import numpy as np
import pickle

from standard.image_process import pre_processing_to_test

resize_width = 60
resize_height = 15

# load model
with open("omr_model.pkl", "rb") as f:
    model = pickle.load(f)

def predict_image(image_path):
    img = cv2.imread(image_path)

    # convert image to feature
    feature = pre_processing_to_test(img)

    prediction = model.predict(feature)

    return prediction[0]


# result = predict_image("D:\Coding\MSE35HN\MLE501\dataset_tracnghiem\A\cau_001_A_000.png")
# result = predict_image("D:\Coding\MSE35HN\MLE501\dataset_tracnghiem\B\cau_001_B_015.png")
# result = predict_image("D:\Coding\MSE35HN\MLE501\dataset_tracnghiem\C\cau_001_C_014.png")
result = predict_image("D:\Coding\MSE35HN\MLE501\dataset_tracnghiem\D\cau_001_D_038.png")

print("Predicted answer:", result)