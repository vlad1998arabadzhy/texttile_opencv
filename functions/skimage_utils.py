import os

from imageio.v2 import imsave
from skimage.color import label2rgb, rgb2gray, rgba2rgb
from skimage.feature import corner_harris, corner_peaks
from skimage.filters import threshold_otsu
from skimage.io import imread
from skimage.measure import label, regionprops, regionprops_table
from skimage.morphology import footprint_rectangle, closing, remove_small_objects
from skimage.segmentation import clear_border
from xgboost import XGBClassifier
import shutil
from functions.area_stat import  *
from functions.check_df import *

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches




"""
This file has functions for image processing and feature extraction
Main functionality func is extract features
To apply main function should be called  traverse_datafolder()
"""



PIXELS_FOLDER="/home/garuda/PycharmProjects/KI_1/pixels" # Folder of the initial input dataset

DATASETS_PATH = "/home/garuda/PycharmProjects/KI_1/processed_img_features/"

# Coefficient for convertion pixels to mm^2
K= 0.0168662169

PROPS=['label', 'area','perimeter','solidity']#TODO has to be updated according to output of extract features


TRASH = "Trash"
TOO_BIG = "Too big"
TOO_SMALL = "Too small"
NORMAL= "Normal"


def add_category_set(model:XGBClassifier, df:pd.DataFrame):
    df['category']=create_category_ser(model,df)


def extract_features(path, path_to_save_image=None):

    #Load
    img = imread(path)
    #shutil.copy(path, path_to_save_image)#TODO needed for creating datasets for Supervised Learning

    img = rgba2rgb(img)

    #Grayscaling
    img = rgb2gray(img)

    thresh = threshold_otsu(img)
    bw=closing(img>thresh, footprint_rectangle((3,3)))

    bw = remove_small_objects(bw, min_size=100)

    cleared =clear_border(bw)

    labeled_image = label(cleared)

    props = regionprops_table(labeled_image,
                        properties=PROPS)

    regions = regionprops(labeled_image)

#<----------------Add Features------------------->

    df = pd.DataFrame(props)
    add_corners_to_df(df,regions)

    add_size(df,path)
    add_size_normalized(df)
    add_corners_normalization_to_df(df)
    add_width(df,labeled_image)
    add_height(df,labeled_image)
    add_perfect_area(df)
    width_is_like_height(df)


    pixels2milimeters(df)
    add_area_normalized(df)
    add_category_set(MODEL, df)#TODO uncomment after training of model
    rename_columns(df)


#<------------------------Display processed image--------------------->

    image_label_overlay = label2rgb(labeled_image, image=img, bg_label=0)
    # fig, ax = plt.subplots(figsize=(10, 6))
    # ax.imshow(image_label_overlay)
    #
    #
    #
    # for region in regionprops(labeled_image):
    #     # take regions with large enough areas
    #     if region.area >= 100:
    #         # draw rectangle around segmented coins
    #         minr, minc, maxr, maxc = region.bbox
    #         height = maxr - minr
    #         width = maxc - minc
    #
    #         rect = mpatches.Rectangle(
    #             (minc, minr),
    #             maxc - minc,
    #             maxr - minr,
    #             fill=False,
    #             edgecolor='red',
    #             linewidth=2,
    #         )
    #         ax.add_patch(rect)
    #         ax.text(
    #             minc,  # x
    #             minr ,  # y (чуть выше прямоугольника)
    #             f"ID {region.label}",  # текст
    #             color='yellow',
    #             fontsize=8,
    #             fontweight='bold',
    #             bbox=dict(
    #                 facecolor='black',
    #                 alpha=0.5,
    #                 edgecolor='none'
    #             )
    #         )
    #
    # ax.set_axis_off()
    # plt.tight_layout()
    #plt.savefig(path_to_save_image+".png", format='png', bbox_inches='tight')# TODO Needed only to create datasets for supervised ML

   # plt.show()



    return df



def add_width(df:pd.DataFrame, labeled_image):
    def calculate_width(region):
        _, minc, _, maxc = region.bbox
        return maxc - minc
    df['width'] = pd.Series([calculate_width(region) for region in regionprops(labeled_image) if region.area >= 100])



def add_height(df:pd.DataFrame, labeled_image):
    def calculate_height(region):
        minr, _, maxr, _ = region.bbox
        return maxr - minr
    df['height'] = pd.Series([calculate_height(region) for region in regionprops(labeled_image) if region.area >= 100])




def count_corners(mask, min_distance=3, threshold_rel=0.01):
    response = corner_harris(mask)
    corners = corner_peaks(
        response,
        min_distance=min_distance,
        threshold_rel=threshold_rel,
    )
    return corners


def pixels2milimeters(df:pd.DataFrame, headers:list=["area","perimeter","height","width"]):
    for header in headers:
        if header == "area":
            df[header]=df[header].apply(lambda x: round(x*K,2))
        else:
            df[header] = df[header].apply(lambda x: round(x * K**(1/2), 2))



def add_corners_to_df(df:pd.DataFrame, regions):
    df["corners"]=df["label"].apply(lambda x: len(count_corners(regions[x-1].image)))

def add_corners_normalization_to_df(df:pd.DataFrame):
    df["corners_normalized"]=df["corners"]/(df["area_size"]*4)

def create_df(path, name:str):
    df = pd.DataFrame(columns=PROPS)
    for file in os.listdir(path):
        if file.endswith(".png") and "mask" not in file:
            df = pd.concat([df, extract_features(path + f"/{file}")], ignore_index=True)

    df.to_csv((DATASETS_PATH+name+".csv"), index=False)
    return df



def width_is_like_height(df:pd.DataFrame):
    df['width_to_height_ratio'] = df["width"]/df["height"]

def rename_columns(df:pd.DataFrame):
    df.rename(columns={"area":AREA_HEADER, "label":OBJ_NUMBER})



def traverse_datafolder():
    for file in os.scandir(PIXELS_FOLDER):
        print(file.path)
        #TODO Creates files from 0 to 9 , not from 1 to 10 in processed_img_features.
        #TODO Fix before using Machine or rename 10.csv to 10.csv manually
        name = file.path[-1]
        create_df(file.path,name )

def add_size(df:pd.DataFrame, path):
    #TODO this returns size 1 for but has to be 100
    size=path.replace(".png","").replace(PIXELS_FOLDER+"/","")
    if size.startswith("1") and not size.startswith("10"):
        size="1"
    elif size.startswith("10"):
        size="10"
    else:
        size=size[0]

    sizes = [int(size)*10 for x in range(len(df))]
    df["area_size"]= pd.Series(sizes)


def add_size_modified( path):
    size=path.replace(".png","").replace((PIXELS_FOLDER+'/'),"")
    print(size)
    if size.startswith("1.png") and not size.startswith("10"):
        size="1.png"
    elif size.startswith("10"):
        size="10"
    else:
        size=size[0]
    #Convert size from cm to mm.
    sizes = [int(size)*10 for x in range(2)]
    return sizes[0]


def add_size_normalized(df:pd.DataFrame):
    df['size_normalized']=df['area']/((df['area_size'])**2)

def add_perfect_area(df:pd.DataFrame):
    df['perfect_area']=(df['area_size'])**2

def add_area_normalized(df:pd.DataFrame):
    df['area_normalized']=df['area']/df['perfect_area']


