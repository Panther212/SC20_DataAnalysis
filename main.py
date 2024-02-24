import streamlit as st
import numpy as np
import google.cloud
from google.cloud import firestore
import pandas as pd
import joblib
from prediction import predict
from google.cloud.firestore import FieldFilter
from sklearn.metrics import accuracy_score, classification_report
from scipy import signal
from sklearn.preprocessing import MinMaxScaler


st.set_page_config(layout="wide")
st.title('Data Analytics')
st.markdown(
    """
    <style>
    .reportview-container {{
        background-color: white;
    }}
    </style>
    """,
    unsafe_allow_html=True,
  )

# Authenticate to Firestore with the JSON account key.
db = firestore.Client.from_service_account_json("testdata1-20ec5-firebase-adminsdk-an9r6-d15c118c96.json")

number = st.text_input('Enter Tree number', '')
 
# Create a reference to the Google post.
#doc_ref = db.collection("T1R1").select("ADXL Raw", "Radar Raw").stream()


doc_ref = db.collection("DevMode").stream()
i=1 
df = pd.DataFrame()
df2 = pd.DataFrame()
df3 = pd.DataFrame()
TreeNos_list = []

query = db.collection('DevMode').where(filter=FieldFilter("TreeNo", "==", 5)).get()

for doc in query:
    adxl = doc.to_dict()['ADXLRaw']
    radar = doc.to_dict()['RadarRaw']
    #true_labels = doc.to_dict()['InfStat']
    df['Radar '+str(i)] = pd.Series(radar)
    df2['ADXL '+str(i)] = pd.Series(adxl)
    TreeNos_list.append(doc.to_dict()['InfStat'])
    i+=1
    
 #   df['Radar'] = pd.DataFrame.from_dict(radar, orient='index');
 #   df['ADXL'] = pd.DataFrame.from_dict(adxl, orient='index');
#    i+=1
df = df.dropna()
df2 = df2.dropna()

#st.write(df);
st.write(df); 
st.write(df2); 
st.write(TreeNos_list)

#TreeNos_list.to_numpy

@st.cache
def convert_df(df):
 return df.to_csv().encode('utf-8')
csv = convert_df(df)
csv2 = convert_df(df2)

st.download_button(
     "Press to Download Radar",
     csv,
     "Radar.csv",
     "text/csv",
     key='download-csvradar'
 )

st.download_button(
     "Press to Download ADXL",
     csv2,
     "ADXL.csv",
     "text/csv",
     key='download-csvadxl'
 )



def calculate_and_transform_statistics_radar(df, group_size):
    result_df = pd.DataFrame()

    for column in df.columns:
        std_list, ptp_list, mean_list, rms_list = [], [], [], []

     #for i in range(0, len(df), group_size):
        #data_subset = df[column].iloc[i:i+group_size]
        dfx= df[column]
        dfx = pd.DataFrame(dfx)
        dfx = dfx.dropna()
        std_value = np.std(df[column])
        ptp_value = np.ptp(dfx)
        mean_value = np.mean(dfx)
        rms_value = np.sqrt(np.mean(dfx**2))

        std_list.append(std_value)
        ptp_list.append(ptp_value)
        mean_list.append(mean_value)
        rms_list.append(rms_value)


        column_result_df = pd.DataFrame({
        "STD": std_list,
        "PTP": ptp_list,
        "Mean": mean_list,
        "RMS": rms_list
        })
        result_df = pd.concat([result_df, column_result_df],axis=0)
    #st.write(result_df)

    #df_melted = pd.melt(result_df, value_vars=['STD', 'PTP', 'Mean', 'RMS'], var_name='Variable', value_name='Value')
    #df_melted['Type'] = df_melted.groupby('Variable').cumcount()
    #df_result = df_melted.pivot_table(index='Type', columns='Variable', values='Value', aggfunc='first').reset_index(drop=True)
    #df_result.columns = ['STD', 'PTP', 'Mean', 'RMS']
    st.write(result_df)
    return result_df



def calculate_and_transform_statistics_adxl(df):
    result_df = pd.DataFrame()

    for column in df.columns:
        std_list, ptp_list, mean_list, rms_list = [], [], [], []

     #for i in range(0, len(df), group_size):
        #data_subset = df[column].iloc[i:i+group_size]
        dfx= df[column]
        dfx = pd.DataFrame(dfx)
        dfx = dfx.dropna()
        std_value = np.std(df[column])
        ptp_value = np.ptp(dfx)
        mean_value = np.mean(dfx)
        rms_value = np.sqrt(np.mean(dfx**2))

        std_list.append(std_value)
        ptp_list.append(ptp_value)
        mean_list.append(mean_value)
        rms_list.append(rms_value)


        column_result_df = pd.DataFrame({
        "STD_ADXL": std_list,
        "PTP_ADXL": ptp_list,
        "Mean_ADXL": mean_list,
        "RMS_ADXL": rms_list
        })
        result_df = pd.concat([result_df, column_result_df],axis=0)

    #df_melted = pd.melt(result_df, value_vars=['STD_ADXL', 'PTP_ADXL', 'Mean_ADXL', 'RMS_ADXL'], var_name='Variable', value_name='Value')
    #df_melted['Type'] = df_melted.groupby('Variable').cumcount()
    #df_result = df_melted.pivot_table(index='Type', columns='Variable', values='Value', aggfunc='first').reset_index(drop=True)
    #df_result.columns = ['STD_ADXL', 'PTP_ADXL', 'Mean_ADXL', 'RMS_ADXL']
    st.write(result_df)
    return result_df


def radar_fq(df):
    frequencies = []
    powers = []

    for i in df:
        f, p = signal.welch(df[i], 100, 'flattop', 1024, scaling='spectrum')
        powers.append(p)
    powers = pd.DataFrame(powers)
    return powers

def adxl_fq(df):
    frequencies = []
    powers = []

    for i in df:
        f, p = signal.welch(df[i], 20, 'flattop', 256, scaling='spectrum')
        powers.append(p)
    powers = pd.DataFrame(powers)
    return powers

def norm(df):
  scaler = MinMaxScaler()
  df_normalized = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
  return df_normalized

def freqnstatdf(df,df2):
   df_radar_result = radar_fq(df)
   df_adxl_result = calculate_and_transform_statistics_adxl(df2)
   df_adxl_result = df_adxl_result.reset_index(drop=True)
   radar_new_column_names = ['radar' + str(i) for i in range(df_radar_result.shape[1])]
   df_radar_result.columns = radar_new_column_names
   df_test = pd.concat([df_adxl_result,df_radar_result],axis=1)

   return df_test
   
def freqdf(df,df2):
   df_radar_result = radar_fq(df)
   df_adxl_result = adxl_fq(df2)
   adxl_new_column_names = ['adxl' + str(i) for i in range(df_adxl_result.shape[1])]
   df_adxl_result.columns = adxl_new_column_names
   radar_new_column_names = ['radar' + str(i) for i in range(df_radar_result.shape[1])]
   df_radar_result.columns = radar_new_column_names
   df_test = pd.concat([df_adxl_result,df_radar_result],axis=1)
   return df_test

def normpdf(df,df2):
   df_radar_result = norm(df)
   df_adxl_result = norm(df2)
   df_radar_result = radar_fq(df_radar_result)
   df_adxl_result = adxl_fq(df_adxl_result)
   adxl_new_column_names = ['adxl' + str(i) for i in range(df_adxl_result.shape[1])]
   df_adxl_result.columns = adxl_new_column_names
   radar_new_column_names = ['Radar' + str(i) for i in range(df_radar_result.shape[1])]
   df_radar_result.columns = radar_new_column_names
   df_test = pd.concat([df_adxl_result,df_radar_result],axis=1)
   return df_test
   
def normstatdf(df,df2):
   df_radar_result = norm(df)
   df_adxl_result = norm(df2)
   df_radar_result = radar_fq(df_radar_result)
   df_adxl_result = calculate_and_transform_statistics_adxl(df_adxl_result)
   df_adxl_result = df_adxl_result.reset_index(drop=True)
   radar_new_column_names = ['Radar' + str(i) for i in range(df_radar_result.shape[1])]
   df_radar_result.columns = radar_new_column_names
   df_test = pd.concat([df_adxl_result,df_radar_result],axis=1)

   return df_test
   
df_freqnstat = freqnstatdf(df,df2) 
df_freq = freqdf(df,df2) 
df_norm_p = normpdf(df,df2) 
df_norm_stat = normstatdf(df,df2) 

if st.button("Run all models"):
 result_clf_freqnstat,result_clf_freq,result_clf_norm_p,result_clf_norm_stat = predict(df_freqnstat,df_freq,df_norm_p,df_norm_stat)
 st.text(result_clf_freqnstat)
 st.text(result_clf_freq)
 st.text(result_clf_norm_p)
 st.text(result_clf_norm_stat)
 result_clf_freqnstat_accuracy = accuracy_score(TreeNos_list,result_clf_freqnstat)
 st.write("result_clf_freqnstat Accuracy = "+str(result_clf_freqnstat_accuracy*100)+"%")
 result_clf_freq_accuracy = accuracy_score(TreeNos_list,result_clf_freq)
 st.write("result_clf_freq Accuracy = "+str(result_clf_freq_accuracy*100)+"%")
 result_clf_norm_p_accuracy = accuracy_score(TreeNos_list,result_clf_norm_p)
 st.write("result_clf_norm_p Accuracy = "+str(result_clf_norm_p_accuracy*100)+"%")
 result_clf_norm_stat_accuracy = accuracy_score(TreeNos_list,result_clf_norm_stat)
 st.write("result_clf_norm_stat Accuracy = "+str(result_clf_norm_stat_accuracy*100)+"%")
