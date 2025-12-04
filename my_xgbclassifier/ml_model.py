import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

first = [1,2,3,1,2,3,1,4]*100
second = ["True","False","False","True","False","False","True","False"]*100
df_without_target = pd.Series(first)
ser_target = pd.Series(second)


X = df_without_target.copy()
y = ser_target.copy()

le = LabelEncoder()
y = le.fit_transform(y)



X_train,X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


model = XGBClassifier(
    n_estimators=300,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="logloss"
)



skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

cv_score = cross_val_score(
    model,
    X,
    y,
    cv=skf,
)

print("Score is: ", cv_score)
print("nMean score is: ", cv_score.mean())




model.fit(X_train, y_train)

test_accuracy = model.score(X_test, y_test)
print("\nTest accuracy is: ", test_accuracy)


y_pred= model.predict(X_test)

class_rep =classification_report(y_test, y_pred, target_names=le.classes_)
print("\nClassidfication report:")

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

def save_model(model):
    print("\nSaving the model:")
    model.save_model('model.xgb')
    print("Model saved")


def load_model(model:XGBClassifier):
    print("\nLoading the model:")
    model.load_model('model.xgb')
    print("Model loaded")


#save_model(model)


