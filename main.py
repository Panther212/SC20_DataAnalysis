import streamlit as st
import numpy as np
import google.cloud
from google.cloud import firestore
import pandas as pd
import joblib
from prediction import predict
from google.cloud.firestore import FieldFilter


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

query = db.collection('DevMode').where(filter=FieldFilter("TreeNo", "==", number)).get()

for doc in query:
    adxl = doc.to_dict()['ADXLRaw']
    radar = doc.to_dict()['RadarRaw']
    df['Radar '+str(i)] = pd.Series(radar)
    df2['ADXL '+str(i)] = pd.Series(adxl)
    i+=1
    
 #   df['Radar'] = pd.DataFrame.from_dict(radar, orient='index');
 #   df['ADXL'] = pd.DataFrame.from_dict(adxl, orient='index');
#    i+=1

#st.write(df);
st.write(df); 
st.write(df2); 


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



def calculate_and_transform_statistics_adxl(df, group_size):
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


df_radar_result = calculate_and_transform_statistics_radar(df, 100)
df_adxl_result  = calculate_and_transform_statistics_adxl(df2, 20)
df_test = pd.concat([df_radar_result,df_adxl_result],axis=1)
df_test = df_test.dropna()

st.write(df_test)


if st.button("Predict type of Scan"):
 result = predict(df_test)
 st.text(result)
