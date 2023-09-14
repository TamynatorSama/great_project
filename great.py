#libraries importation
import numpy as np
import pandas as pd
import streamlit as st 
import joblib
from sklearn.preprocessing import StandardScaler


#load the model
model = joblib.load("credit_card_model_new.joblib")
# rnn_model = load_model('C:/Users/Abu Roslaan/Desktop/Research/Alzheimer_Disease/Implementation/FrontEnd/cnn_model.h5')


def transformFunc(n):
  return 'Normal' if n==0 else 'Fraud'
# def run_detection(input):
#     detection = model.predict(input)
#     st.info(detection)


#dictionary to store each label
#title of the application
st.header('Credit card theft detection')

#set medium for image upload
dataset = st.file_uploader("Upload your Data ...", type=["csv", "xlsx"])
submit = st.button('Detect')


#prediction
if submit:
    #check if image uploaded isn't null
    if dataset is not None:
        
        data_frame = pd.read_csv(dataset)
        st.info('Data')
        st.dataframe(data_frame)
        sc = StandardScaler()
        data_frame = data_frame.drop(['Time'],axis=1)
        data_frame['Amount']=sc.fit_transform(pd.DataFrame(data_frame['Amount']))
        # if 'Class' in data_frame.columns:
        X = pd.DataFrame(data_frame.drop('Class',axis=1))
        y = data_frame['Class']
        detection = model.predict(X)
        st.title("Result")
        
        df = pd.DataFrame(map(transformFunc,detection))
        st.info('Result')
        st.dataframe(df)
        # else:
        #     run_detection(data_frame)
        
        # #display result
        # if result in disease_dict:
        # st.info()
        # else:
        #     st.info('I have no clue about this patient')