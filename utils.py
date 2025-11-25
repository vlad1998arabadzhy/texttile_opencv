from pathlib import Path

import pandas as pd
import os
import threading

from anyio import sleep

PIXEL_HEADER="Area (pixels)"

K= 4210.72/249654

paths = [f"/home/garuda/PycharmProjects/KI_1/pixels/{x}" for x in range(2, 11)]



def create_all_csv():
    for i in range(2,(len(paths)+2)):
        print(f"Processing folder {i} : {paths[i-2]}")
        merge_datasets(paths[i-2], f"{i}")






def calculate_area_from_px(pixels):
    return K*pixels


def is_dataset(name:str):
    return name.endswith(".xlsx")


def merge_datasets(path_to_df:str, df_number):
    names = ["Index","Object Number", "Area (pixels)"]
    new_path=f"/home/garuda/PycharmProjects/KI_1/datasets/{df_number}"

    directory = Path(path_to_df)
    for file in os.listdir(directory):
        if is_dataset(file):
            pd.read_excel(directory / file).to_csv(f"{new_path}.csv", header=False,mode="a")
    df = pd.read_csv(f"{new_path}.csv", header=None, names=names)
    set_area(df)
    drop_irrelevant_columns(df)
    create_indexation(df)
    # Save the updated CSV file
    df.to_csv(f"{new_path}_updated.csv", index=False)
    print(f"Document {path_to_df}_updated.csv created")

def set_area(df:pd.DataFrame):
    ls = df[PIXEL_HEADER]
    new_column=[]
    for i in ls:
        new_column.append(round(i*K,2))
    df["Area mm^2"]=new_column

def drop_irrelevant_columns(df:pd.DataFrame):
    df.drop(["Area (pixels)"], axis=1, inplace=True)
    df.drop(["Object Number"], axis=1, inplace=True)
    df.drop(["Index"], axis=1, inplace=True)

def create_indexation(df:pd.DataFrame):
    ls = [x for x in range(1,(len(df))+1)]
    df["Index"]=ls


def create_new_dataset(filename:str):
    df = pd.read_excel(filename)
