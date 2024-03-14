import joblib


def predict(m1,m2):
 model1 = joblib.load("model1_nodet.sav")
 model2 = joblib.load("model2_det.sav")
 #model3 = joblib.load("Model3.sav")
 #lf_norm_stat = joblib.load("norm_stats_clf.sav")
 return model1.predict(m1),model2.predict(m2)
