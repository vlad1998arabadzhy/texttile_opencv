from skimage_utils import *


def classify_pieces(df:pd.DataFrame):
    ser = []
    for index, row in df.iterrows():
        if is_trash():
            ser.append(TRASH)
        elif is_too_big():
            ser.append(TOO_BIG)
        elif is_too_small():
            ser.append(TOO_SMALL)
        elif is_normal():
            ser.append(NORMAL)
    df['clssification']=pd.Series(ser)




def df_checker(func):
    def wrapper(df:pd.DataFrame,column):
        for index, row in df.iterrows():
            perimeter = row["perimeter"]
            size_normalized = row['size_normalized']
            corners_normalized = row['corners_normalized']
            width_to_height_ratio = row['width_to_height_ratio']
            area_normalized =row['area_normalized']
            return func(perimeter, size_normalized, corners_normalized, width_to_height_ratio, area_normalized)

@df_checker
def is_too_small(perimeter, size_normalized, corners_normalized, width_to_height_ratio, area_normalized):
    return area_normalized < 0.70
@df_checker
def is_too_big(perimeter, size_normalized, corners_normalized, width_to_height_ratio, area_normalized):
    return area_normalized > 1.25
@df_checker
def is_trash(perimeter, size_normalized, corners_normalized, width_to_height_ratio, area_normalized):
    return
@df_checker
def is_normal(perimeter, size_normalized, corners_normalized, width_to_height_ratio, area_normalized):
    return
