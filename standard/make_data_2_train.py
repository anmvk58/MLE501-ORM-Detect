import os
import cv2

from standard.image_process import convert_img_to_bw, remove_vertical_line, remove_and_get_number_question, \
    remove_horizontal_lines

DATA_PATH = r"D:\Coding\MSE35HN\MLE501\dataset_tracnghiem\D"

SAVE_PATH = r"D:\Coding\MSE35HN\MLE501\data_cleaned\D"

def process_one_image(img_path, save_path):
    try:
        image = cv2.imread(img_path)
        thresh_img = convert_img_to_bw(image)
        cropped_img = remove_vertical_line(thresh_img)
        clean_img = remove_horizontal_lines(cropped_img)
        # only take answer_img
        answer_img = remove_and_get_number_question(clean_img)[1]
        cv2.imwrite(save_path, answer_img)
        print(f"Done: {save_path}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == '__main__':
    for filename in os.listdir(DATA_PATH):
        if filename.endswith(".png"):
            file_path = os.path.join(DATA_PATH, filename)
            save_path = os.path.join(SAVE_PATH, 'train_' + filename)
            process_one_image(img_path=file_path, save_path=save_path)
    # process_one_image('D:\Coding\MSE35HN\MLE501\orm_detect\cau_120_A_018.png', 'test.png')
    # process_one_image('D:\Coding\MSE35HN\MLE501\orm_detect\cau_120_A_018.png', 'test.png')

