"""
OMR Bubble Detection
--------------------
Nhận diện đáp án A/B/C/D từ ảnh trắc nghiệm.

Yêu cầu:
pip install opencv-python numpy

Cách chạy:
python omr_detect.py path_to_image
"""

import cv2
import numpy as np
import sys


def preprocess_image(image):
    """
    Tiền xử lý ảnh:
    - chuyển sang grayscale
    - blur để giảm noise
    - threshold để tạo ảnh trắng đen
    """

    # chuyển sang grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # làm mịn ảnh để giảm nhiễu
    blur = cv2.GaussianBlur(gray, (5,5), 0)

    # threshold để tách vùng đen trắng
    thresh = cv2.threshold(
        blur,
        150,
        255,
        cv2.THRESH_BINARY
    )[1]

    return thresh


def find_bubble_contours(thresh):
    """
    Tìm contour trong ảnh và lọc ra các contour
    có hình dạng gần giống bubble tròn.
    """

    contours,_ = cv2.findContours(
        thresh,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    bubble_contours = []

    for c in contours:

        # bounding box của contour
        x,y,w,h = cv2.boundingRect(c)

        # diện tích contour
        area = cv2.contourArea(c)

        # tỉ lệ width/height
        ratio = w / float(h)

        # điều kiện để coi là bubble
        if area > 150 and 0.8 <= ratio <= 1.2:

            bubble_contours.append(c)

    return bubble_contours


def sort_bubbles_left_to_right(bubble_contours):
    """
    Sắp xếp bubble theo thứ tự từ trái qua phải.
    """

    bubble_contours = sorted(
        bubble_contours,
        key=lambda c: cv2.boundingRect(c)[0]
    )

    return bubble_contours


def compute_bubble_scores(thresh, bubbles):
    """
    Tính độ đậm (pixel đen) của từng bubble.
    Bubble được tô sẽ có nhiều pixel đen hơn.
    """

    scores = []

    for c in bubbles:

        x,y,w,h = cv2.boundingRect(c)

        # crop vùng bubble
        bubble = thresh[y:y+h, x:x+w]

        # đếm pixel đen
        total = np.sum(bubble)

        scores.append(total)

    return scores


def detect_answer(image_path):
    """
    Hàm chính để detect đáp án từ ảnh.
    """

    # đọc ảnh
    image = cv2.imread(image_path)

    if image is None:
        print("Không đọc được ảnh.")
        return None

    # resize để ổn định pipeline
    image = cv2.resize(image, (365,47))

    # tiền xử lý
    thresh = preprocess_image(image)

    # tìm bubble contour
    bubbles = find_bubble_contours(thresh)

    if len(bubbles) < 4:
        print("Không phát hiện đủ bubble.")
        return None

    # sắp xếp bubble theo trục ngang
    bubbles = sort_bubbles_left_to_right(bubbles)

    # chỉ giữ 4 bubble đầu tiên
    bubbles = bubbles[:4]

    # tính score của từng bubble
    scores = compute_bubble_scores(thresh, bubbles)

    # mapping đáp án
    answers = ['A','B','C','D']

    # bubble có score lớn nhất
    index = np.argmax(scores)

    answer = answers[index]

    return answer, scores


def main():

    if len(sys.argv) < 2:
        print("Usage: python omr_detect.py image_path")
        return

    image_path = sys.argv[1]

    result = detect_answer(image_path)

    if result is None:
        return

    answer, scores = result

    print("Bubble scores:", scores)
    print("Detected answer:", answer)


if __name__ == "__main__":
    main()