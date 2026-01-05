from ml_model import  load_model, X_train, y_train
from sklearn.metrics import  accuracy_score

#Here we tested if the trained model was correctly saved

model = load_model("/home/garuda/PycharmProjects/KI_1/my_xgbclassifier/model_v1.xgb")

y_pred = model.predict(X_train)

#Result: 94% of accuracy. Model saved perfectly.
print("Accuracy: ", accuracy_score(y_train, y_pred))
