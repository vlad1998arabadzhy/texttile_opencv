import pandas as pd
import  os


PATHS_OF_TRAINING_DATA ="/home/garuda/PycharmProjects/KI_1/datasets_to_supervise/"

def concatenate_training_datasets():
    """
    Creates one big .csv from datasets_to_supervise folder
    TODO activate this function , after proccesing datasets manually
    :return:
    """
    columns = pd.read_csv("/home/garuda/PycharmProjects/KI_1/datasets_to_supervise/1/1.csv").columns;
    df = pd.DataFrame(columns=columns)
    for directory  in os.listdir(PATHS_OF_TRAINING_DATA):
        for file in os.listdir(PATHS_OF_TRAINING_DATA+directory):
            if file.endswith(".csv"):
                print("Processing " + directory + "/" + file)
                temporal_df = pd.read_csv(PATHS_OF_TRAINING_DATA+directory+"/"+file)
                df= pd.concat([df,temporal_df])

    df.to_csv('/home/garuda/PycharmProjects/KI_1/my_xgbclassifier/training_data/training_data.csv')