import sys

import cv2
import numpy as np
import pytesseract

from standard.image_process import pre_processing_to_view, convert_img_to_bw

np.set_printoptions(threshold=sys.maxsize)
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# def pre_processing_to_view(img):
#     denoised_img = denoise_rayleigh(img)
#     thresh_img = convert_img_to_bw(denoised_img)
#     cropped_img = remove_vertical_line(thresh_img)
#     clean_img = remove_horizontal_lines(cropped_img)
#
#     # get result
#     number_img, answer_img = remove_and_get_number_question(clean_img)
#     return answer_img

if __name__ == '__main__':
    # image = cv2.imread('images.jpg')
    # image = cv2.imread('cau_120_A_018.png')
    # image = cv2.imread('cau_118_A_016.png')
    image = cv2.imread('D:\Coding\MSE35HN\MLE501\dataset_tracnghiem\A\cau_001_A_008.png')
    # image = cv2.imread('cau_001.png')
    denoised_img = convert_img_to_bw(image)

    answer, number = pre_processing_to_view(image)

    cv2.imshow('Answer', answer)
    cv2.imshow('Number', number)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
