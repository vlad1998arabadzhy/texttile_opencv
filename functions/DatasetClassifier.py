from pprint import pprint
import  os
import pandas as pd
from xgboost import XGBClassifier

from functions.skimage_utils import PROPS
from my_xgbclassifier.ml_model import load_model


class DatasetClassifier:
    """
    This class adds labels(1, 2, 3, 4) to the existing datasets
    and merges them to the one  .csv-file
    """
    def __init__(self, model:XGBClassifier, path_to_sets:str):
        self.model= model
        self.path_to_sets=path_to_sets
        self.paths =[f"{path_to_sets}/{x}.csv"for x in range(1,11)]

    def get_paths(self):
        """
        Test function to be sure that paths are correct
        :return:
        """
        return self.paths

    def classify_set(self, path:str):
        """
        Label a dataset on respective path.
        :param path: path to the data set

        """
        pass

    def classify_all_sets(self):
        """
         Label all datasets.
        """
        for path in self.paths:
            self.classify_set(path)
    def merge_all_data_sets(self, ):
        """
        Merges all .csv into one
        """

        df = pd.DataFrame(columns=PROPS)
        for file in os.listdir(self.path_to_sets):
            if file.endswith(".csv"):
                temporal_df= pd.read_csv(file)
                df = pd.concat([df, temporal_df], ignore_index=True)
        df.to_csv("/home/garuda/PycharmProjects/KI_1/final_result.csv", index=False)
        return df
        pass


PROPS.append("label")
#model = load_model("/home/garuda/PycharmProjects/KI_1/functions/model_v1.xgb")
labeler = DatasetClassifier(None, "/home/garuda/PycharmProjects/KI_1/processed_img_features")
pprint(labeler.get_paths())
