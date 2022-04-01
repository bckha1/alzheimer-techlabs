import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import train_test_split
import streamlit as st
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score,confusion_matrix
import io


def load_model():
    # load the model from disk
    model = pickle.load(open("finalized_random-forest.sav", 'rb'))
    return(model)


def data_reader():
    # following lines create boxes in which user can enter data required to make prediction 
    MMSE = st.number_input('MMSE')
    CDR = st.number_input("CDR") 
    return(MMSE,CDR)
    
def result(MMSE,CDR):
    model=load_model()
    prediction = model.predict([[MMSE,CDR]])
    # Making predictions 
    prediction1 = model.predict_proba([[MMSE,CDR]])
    pred="The proba of developing dementia for this patient is: "+prediction1[0][1]
    return pred


def main():
       MMSE,CDR=data_reader()
       # when 'Predict' is clicked, make the prediction and store it 
       if st.button("Predict"): 
           pred = result(MMSE,CDR)
           st.write(pred)
   
main()
