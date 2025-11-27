import cv2
import numpy as np
import pandas as pd

MASK_PATH ="/home/garuda/PycharmProjects/KI_1/pixels/1/1.7.png"

TEST_PATH="/home/garuda/PycharmProjects/KI_1/notebooks/mask_binary.png"
mask = cv2.imread(MASK_PATH, 0)

countours, hierachy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
obj_quantity = len(countours)


def extract_all_features(mask_path):
    mask = cv2.imread(mask_path)
    gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)

    countours, _ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    row =[]
    for cnt in countours:
        features = extract_features(cnt)
        row.append(features)
    return pd.DataFrame(row)



def extract_features(cnt):


    area = cv2.contourArea(cnt)
    x,y,w,h = cv2.boundingRect(cnt)# Rectangle of my masc

    hull = cv2.convexHull(cnt)
    hull_area = cv2.contourArea(hull)

    aspect = w/h
    extend = area / aspect
    solidity  = area / hull_area
    perimeter = cv2.arcLength(cnt,True)

    #Ramer-Douglas-Algo to simplify counter
    epsilon = 0.03*cv2.arcLength(cnt,True)
    approx = cv2.approxPolyDP(cnt, epsilon, True )
    vertices = len(approx)
    return {
        "area_px":area,
        "perimeter":perimeter,
        "solidity":solidity,
        "extent":extend,
        "aspect":aspect,
        "vertices":vertices,
        "width":w,
        "height":h
    }


def draw_object_numbers(mask, contours):
    img_color = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

    for i, cnt in enumerate(contours):
        x, y, w, h = cv2.boundingRect(cnt)


        tx = x
        ty = y - 5 if y > 20 else y + 20

        txt = str(i + 1)


        cv2.putText(img_color, txt, (tx, ty),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 0, 0), 4)

        cv2.putText(img_color, txt, (tx, ty),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (255, 255, 255), 2)

    return img_color


def create_mask(mask_path):
    # Read directly as grayscale
    gray = cv2.imread(mask_path, 0)

    # Blur to remove noise
    blur = cv2.GaussianBlur(gray, (5,5), 0)

    # Otsu threshold
    _, mask = cv2.threshold(
        blur,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    # Morphology to clean the shape
    kernel = np.ones((5,5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    cv2.imwrite("mask_binary.png", mask)

    return mask