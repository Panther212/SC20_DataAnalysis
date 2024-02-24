import joblib


def predict(df_freqnstat,df_freq,df_norm_p,df_norm_stat):
 clf_freqnstat = joblib.load("freq_n_stats_clf.sav")
 clf_freq = joblib.load("freq_clf_.sav")
 clf_norm_p = joblib.load("norm_p_clf.sav")
 clf_norm_stat = joblib.load("norm_stats_clf.sav")
 return clf_freqnstat.predict(df_freqnstat), clf_freq.predict(df_freq),clf_norm_p.predict(df_norm_p), clf_norm_stat.predict(df_norm_stat)
