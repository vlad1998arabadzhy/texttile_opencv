import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

df = pd.read_csv("/home/garuda/PycharmProjects/KI_1/my_xgbclassifier/training_data/training_data.csv")

# Убираем мусор из обучения
df_clean = df[df["category"] != 0]

X = df_clean.drop(["category"], axis=1)
y = df_clean["category"]

le = LabelEncoder()
y = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = XGBClassifier(
    n_estimators=300,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="logloss"
)

# Cross-validation
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_score = cross_val_score(model, X, y, cv=skf)

print("Score is:", cv_score)
print("Mean score is:", cv_score.mean())

model.fit(X_train, y_train)
print("\nTest accuracy:", model.score(X_test, y_test))

# Confusion matrix для классов 1,2,3
y_pred = model.predict(X_test)
print("\nConfusion Matrix (classes 1,2,3 only):")
print(confusion_matrix(y_test, y_pred))
print("\n", classification_report(y_test, y_pred))
def save_model(model):
    print("\nSaving the model:")
    model.save_model('model_v1.xgb')
    print("Model saved")


def load_model(path:str)-> XGBClassifier:
    model = XGBClassifier()
    print("\nLoading the model:")
    model.load_model(path)
    print("\nModel was loaded")
    return model





save_model(model)