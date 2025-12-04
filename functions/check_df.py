from xgboost import XGBClassifier
from my_xgbclassifier.ml_model import  *
from functions.skimage_utils import *

MODEL = XGBClassifier()#load_model("/home/garuda/PycharmProjects/KI_1/my_xgbclassifier/model.xgb") #TODO when model is ready -- change


def predict_category(model, df: pd.DataFrame, index: int):
    row = df.iloc[index]
    return model.predict(row)


def create_category_ser(model: XGBClassifier, df: pd.DataFrame)->pd.Series:
    ans = [predict_category(model, df, i)[0] for i in range(len(df))]

    return pd.Series(ans)