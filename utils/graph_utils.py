import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from utils.area_stat import *
import scipy.stats as stats
EXAMPLE_PATH="/home/garuda/PycharmProjects/KI_1/area_sets/4_updated.csv"

DF = pd.read_csv(EXAMPLE_PATH)
ser = DF[AREA_HEADER]
FILE_NAMES= [f"/home/garuda/PycharmProjects/KI_1/area_sets/{x}_updated.csv" for x in range(1,11)]



def create_graph(ser:pd.Series, size:str):
    plt.figure(figsize=(10,5))
    #My data

    if size == "1x1":
        plt.xlim([0,600])
        plt.hist(ser, bins=200, density=True, alpha=0.5, color='skyblue')

    elif size == "2x2":
        plt.xlim([0, 1000])
        plt.hist(ser, bins=100, density=True, alpha=0.5, color='skyblue')
    elif size == "3x3":
        plt.xlim([0, 2000])
        plt.hist(ser, bins=100, density=True, alpha=0.5, color='skyblue')

    else:
        plt.hist(ser, bins=50, density=True, alpha=0.5, color='skyblue')


    plt.xlabel("Square area mm**2")
    plt.ylabel("Frequency")
    plt.title(size)


    #Normal distribution
    mean= ser.mean()
    std= ser.std()

    x_nd= np.linspace(ser.min(),ser.max(),ser.count())
    pdf =stats.norm.pdf(x_nd,mean,std)

    # Create plot
    plt.plot(x_nd,pdf, 'r-', linewidth=2)
    #plt.savefig(f"/img/{size}_graph.png")
    plt.show()



def extract_all_series(file_names):
    ls = []
    for file in file_names:

        name = (file
                .replace("_updated.csv","")
                .replace("/home/garuda/PycharmProjects/KI_1/area_sets/",""))
        name = name + 'x'+name
        ls.append((name,pd.read_csv(file)[AREA_HEADER]))
    return ls




def create_all_graphs(series):
    for ser, name in series:
        create_graph(name, ser)