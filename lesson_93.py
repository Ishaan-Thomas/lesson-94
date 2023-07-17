# S2.1: Open Sublime text editor, create a new Python file, copy the following code in it and save it as 'improved_iris_app.py'.
# Importing the necessary libraries.
import numpy as np
import streamlit as st
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split as tts
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
# Loading the dataset.
df=pd.read_csv('iris-species.csv')

# Adding a column in the Iris DataFrame to resemble the non-numeric 'Species' column as numeric using the 'map()' function.
# Creating the numeric target column 'Label' to 'iris_df' using the 'map()' function.
df['label']=df['Species'].map({'Iris-setosa':0,'Iris-virginica':1,'Iris-versicolor':2})

# Creating features and target DataFrames.
X=df[['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']]
y=df['label']
# Splitting the dataset into train and test sets.
X_train,X_test,y_train,y_test=tts(X,y,random_state=42,test_size=.3)

# Creating an SVC model.
svc=SVC(kernel='linear')
svc.fit(X_train,y_train)
# Creating a Logistic Regression model.
lr=LogisticRegression(n_jobs=-1)
lr.fit(X_train,y_train)
# Creating a Random Forest Classifier model.
rfc=RandomForestClassifier(n_jobs=-1,n_estimators=100)
rfc.fit(X_train,y_train)


# S2.2: Copy the following code in the 'improved_iris_app.py' file after the previous code.
# Create a function that accepts an ML mode object say 'model' and the four features of an Iris flower as inputs and returns its name.
@st.cache()
def prediction(model,sepal_length,sepal_width,petal_length,petal_width):
  species=model.predict([[sepal_length,sepal_width,petal_length,petal_width]])
  species=species[0]
  if species==0:
    return 'Iris-setosa'
  elif species==1:
    return 'Iris-virginica'
  else:
    return 'Iris-versicolor'



# S2.3: Copy the following code and paste it in the 'improved_iris_app.py' file after the previous code.
# Add title widget
st.sidebar.title('Iris Flower Species Prediction App')

# Add 4 sliders and store the value returned by them in 4 separate variables.
s_len=st.sidebar.slider('Sepal length',float(df['SepalLengthCm'].min()),float(df['SepalLengthCm'].max()))
# The 'float()' function converts the 'numpy.float' values to Python float values.
s_width=st.sidebar.slider('Sepal width',float(df['SepalWidthCm'].min()),float(df['SepalWidthCm'].max()))
p_len=st.sidebar.slider('Petal length',float(df['PetalLengthCm'].min()),float(df['PetalLengthCm'].max()))
p_width=st.sidebar.slider('Petal width',float(df['PetalWidthCm'].min()),float(df['PetalWidthCm'].max()))
# Add a select box in the sidebar with the 'Classifier' label.
# Also pass 3 options as a tuple ('Support Vector Machine', 'Logistic Regression', 'Random Forest Classifier').
# Store the current value of this slider in the 'classifier' variable.
classifier=st.sidebar.selectbox('Classifier',('Support Vector Machine', 'Logistic Regression', 'Random Forest Classifier'))

# When the 'Predict' button is clicked, check which classifier is chosen and call the 'prediction()' function.
# Store the predicted value in the 'species_type' variable accuracy score of the model in the 'score' variable.
# Print the values of 'species_type' and 'score' variables using the 'st.text()' function.
if st.sidebar.button('Predict'):
  if classifier=='Support Vector Machine':
    species_type=prediction(svc,s_len,s_width,p_len,p_width)
    score=svc.score(X_train,y_train)
  elif classifier=='Logistic Regression':
    species_type=prediction(lr,s_len,s_width,p_len,p_width)
    score=lr.score(X_train,y_train)
  else:
    species_type=prediction(rfc,s_len,s_width,p_len,p_width)
    score=rfc.score(X_train,y_train)
  st.write('Species Predicted',species_type)
  st.write('Accuracy score of this model',score)