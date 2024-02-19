import streamlit as st
import numpy as np
import google.cloud
from google.cloud import firestore
import pandas as pd

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

number = st.text_input('Enter Scan number', '')
 
# Create a reference to the Google post.
#doc_ref = db.collection("T1R1").select("ADXL Raw", "Radar Raw").stream()


doc_ref = db.collection("T1R1").stream()
i=1 
for doc in doc_ref:
    adxl = u'{}'.format(doc.to_dict()['ADXL Raw'])
    radar = u'{}'.format(doc.to_dict()['Radar Raw'])
    df['Radar '+i] = pd.DataFrame.from_dict(radar, orient='index');
    df['ADXL '+i] = pd.DataFrame.from_dict(adxl, orient='index');
    i++

st.write(df);
 
# Then get the data at that reference.
#doc = doc_ref.get()
#bal = u'{}'.format(doc_ref.to_dict()['Balance'])
#for doc in doc_ref:
# st.write(doc.to_dict())
#Np_result = np.array(doc.to_dict())
#Np_result
#st.write(doc.to_dict())

#df = pd.DataFrame(doc.to_dict())
#st.write(doc.to_dict())
#df = pd.DataFrame.from_dict(doc.to_dict(), orient='index')
#st.write(df); 
#df = df.transpose()
#st.write(df);

#df2 = df[["adxl_Rawdata "+number, "Radar_Rawdata "+number]]
#st.write(df2);
'''
@st.cache
def convert_df(df2):
 return df2.to_csv().encode('utf-8')
csv = convert_df(df2)

st.download_button(
     "Press to Download",
     csv,
     "file.csv",
     "text/csv",
     key='download-csv'
 )
'''
