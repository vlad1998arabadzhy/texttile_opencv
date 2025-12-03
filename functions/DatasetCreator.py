from skimage_utils import *
import pandas as pd

PATH_TO_SAVE = "/home/garuda/PycharmProjects/KI_1/datasets_to_supervise/datasets2/"
paths = [f'/home/garuda/PycharmProjects/KI_1/pixels/{x}/{x}.2.png' for x in range(1,11)]

class DatasetCreator:
    def create_datasets(self, path_to_df:list):
        i=1
        for  path in paths:
            self.create_dataset_for_supervised_ml(path, PATH_TO_SAVE, str(i))
            i+=1

    def create_dataset_for_supervised_ml(self, path_of_data, path_to_save, index):
        df= extract_features(path_of_data, path_to_save+index)
        ser = pd.Series(["NO_CATEGORY" for x in range(len(df))])
        df['category'] = ser
        df.to_csv((path_to_save+index+".csv"), index=False)

dataset = DatasetCreator()
dataset.create_datasets(paths)