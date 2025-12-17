import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from data_visualisation.GraphConfig import GraphConfig
from functions.area_stat import *
import scipy.stats as stats
EXAMPLE_PATH= "/area_sets/4_updated.csv"


FILE_NAMES= [f"/home/garuda/PycharmProjects/KI_1/area_sets/{x}_updated.csv" for x in range(1,11)]


class GraphManager:
    def __create_graph(self,df:pd.DataFrame, target, title:str, xlabel:str,
                     _1x1_restriction:int,
                     _2x2_restriction:int, _3x3_restriction:int):

        plt.figure(figsize=(10,5))
        ser = df[target]
        #My training_data

        if  "1x1" in title:
            plt.xlim([0,_1x1_restriction])
            plt.hist(ser, bins=200, density=True, alpha=0.5, color='skyblue')

        elif "2x2" in title:
            plt.xlim([0, _2x2_restriction])
            plt.hist(ser, bins=100, density=True, alpha=0.5, color='skyblue')
        elif "3x3" in title:
            plt.xlim([0, _3x3_restriction])
            plt.hist(ser, bins=100, density=True, alpha=0.5, color='skyblue')

        else:
            plt.hist(ser, bins=50, density=True, alpha=0.5, color='skyblue')


        plt.xlabel(xlabel)
        plt.ylabel("Frequency")
        plt.title(title)


        #Normal distribution
        mean= ser.mean()
        std= ser.std()

        x_nd= np.linspace(ser.min(),ser.max(),ser.count())
        pdf =stats.norm.pdf(x_nd,mean,std)

        # Create plot
        plt.plot(x_nd,pdf, 'r-', linewidth=2)
        plt.savefig(f"/home/garuda/PycharmProjects/KI_1/notebooks/img/{title}_graph.png")
        plt.show()


    def create_all_graphs(self,paths:list, graph_config:GraphConfig):

        for path in paths:
            self.__create_graph_from_path( path=path, grap_config=graph_config)





    def __create_graph_from_path(self,path:str, grap_config:GraphConfig):
        df = pd.read_csv(path)
        self.__create_graph(target=grap_config.target, title=grap_config.title, xlabel=grap_config.xlabel, df=df,
                     _1x1_restriction=grap_config.x1x1,
                     _2x2_restriction=grap_config.x2x2,
                     _3x3_restriction=grap_config.x3x3)
