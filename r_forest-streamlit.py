import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import train_test_split
import streamlit as st
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score,confusion_matrix
import io


def data_reader(data_path):
    data=pd.read_csv(data_path)
    data.dropna(inplace=True)

    # replacing values
    data['Group'].replace(['Nondemented', 'Demented'],[0, 1], inplace=True)
    data['M/F'].replace(['M', 'F'],[0, 1], inplace=True)
    data=data.drop(data[data["Group"]=="Converted"].index)
    return(data)

def features(data):
    if st.button('Data info'):
        st.header("Data info")
        buffer = io.StringIO()
        data.info(buf=buffer)
        s = buffer.getvalue()
        st.text(s)
    ## split train / test
    cols=list(data.columns)
    x_train,x_test,y_train,y_test = train_test_split(data[cols[1:]],data[cols[0:1]], train_size=0.8, test_size=0.2, shuffle=False)

    y_train=y_train.astype('int')
    y_test=y_test.astype('int')

    y_train=list(y_train["Group"])
    y_test=list(y_test["Group"])

    model=RandomForestClassifier()
    model.fit(x_train,y_train)

    feature_imp = pd.Series(model.feature_importances_,index=cols[1:]).sort_values(ascending=False)

    if st.button('Show most important features'):
        # Creating a bar plot
        c=plt.figure(figsize=(15,15))

        sns.barplot(x=feature_imp, y=feature_imp.index)
        # Add labels to your graph
        plt.xlabel('Feature Importance Score')
        plt.ylabel('Features')
        plt.title("Importantance of Features")
        plt.legend()

        st.header("Significant features")
        st.pyplot(c)    
        st.write("The most important features are MMSE and CDR")


    corr = data.corr()

    if st.button('Show features correlation matrix'):
        c1=plt.figure(figsize=(14,8))
        sns.heatmap(corr,cmap="Blues",annot=True,xticklabels=corr.columns,yticklabels=corr.columns)
        st.header("Heatmap")
        st.pyplot(c1)

def training(data):
    ## split train / test
    cols=list(data.columns)
    x_train,x_test,y_train,y_test = train_test_split(data[["MMSE","CDR"]],data[cols[0:1]], train_size=0.8, test_size=0.2, shuffle=False)
    
    y_train=y_train.astype('int')
    y_test=y_test.astype('int')
    
    y_train=list(y_train["Group"])
    y_test=list(y_test["Group"])

    if st.button('Run model'):
        model=RandomForestClassifier()
        model.fit(x_train,y_train)
        y_pred=model.predict(x_test)


        score=accuracy_score(y_pred,y_test)
        mat=confusion_matrix(y_pred,y_test)

        st.header("Accuracy score")
        st.text(score)
        st.header("Confusion matrix")
        st.write(mat)


def main(data_path):
    if data_path is None:
        st.warning("Please upload a csv file")
    else:
        data=data_reader(data_path)
        features(data)
        training(data)
    
    

data_path=st.file_uploader("Please upload data", type=["csv"])
#data_path = Path(__file__).parents[1] /"alzheimer.csv"
main(data_path)
