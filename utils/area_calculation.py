from pathlib import Path

import pandas as pd
import os
import threading

from anyio import sleep

#TODO Obsolete file.  To delete



PIXEL_HEADER="Area (pixels)"

K= 4210.72/249654

paths = [f"/home/garuda/PycharmProjects/KI_1/pixels/{x}" for x in range(1, 11)]



def create_all_csv():
    for i in range(0,(len(paths))):
        print(f"Processing folder {i+1} : {paths[i]}")
        merge_datasets(paths[i], f"{i+1}")

def calculate_area_from_px(pixels):
    return K*pixels


def is_dataset(name:str):
    return name.endswith(".xlsx")


def merge_datasets(path_to_df:str, df_number):
    names = ["Index","Object Number", "Area (pixels)"]
    new_path=f"/home/garuda/PycharmProjects/KI_1/area_sets/{df_number}"

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
    df["Area mm^2.png"]=new_column

def drop_irrelevant_columns(df:pd.DataFrame):
    df.drop(["Area (pixels)"], axis=1, inplace=True)
    df.drop(["label"], axis=1, inplace=True)
    df.drop(["solidity"], axis=1, inplace=True)

def create_indexation(df:pd.DataFrame):
    ls = [x for x in range(1,(len(df))+1)]
    df["Index"]=ls

