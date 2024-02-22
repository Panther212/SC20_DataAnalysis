import joblib


def predict(data):
 clf = joblib.load("new_data_model.sav")
 return clf.predict(data)
