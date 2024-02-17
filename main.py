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
doc_ref = db.collection("T1R1").document("Scan "+number)
 
# Then get the data at that reference.
doc = doc_ref.get()

Np_result = np.array(doc.to_dict())
Np_result
st.write(doc.to_dict())

'''
df = pd.DataFrame(Np_result)
st.write(df);
@st.cache
def convert_df(df):
 return df.to_csv().encode('utf-8')
csv = convert_df(df)

st.download_button(
     "Press to Download",
     csv,
     "file.csv",
     "text/csv",
     key='download-csv'
 )
 '''
