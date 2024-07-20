import streamlit as st
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# Load the data
data = pd.read_csv(r"Diabetes.csv")

# Title and description
st.title('Diabetes Prediction App')
st.markdown("""
This app predicts the likelihood of diabetes using a Logistic Regression model. 
Please adjust the parameters on the sidebar to see the prediction results.
""")

# Adding an image to the main section
st.image('images (1).jpeg', caption='Diabetes Awareness')

# Subheader for data
st.subheader('Diabetes Dataset Overview')
st.write(data.head())

# Separate the features and the target variable
X = data.drop('Outcome', axis=1)
y = data['Outcome']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train the Logistic Regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Sidebar header
st.sidebar.header('User Input Parameters')

# Adding an image to the sidebar
st.sidebar.image('banner6.jpg', caption='Healthy Living')

def user_input_features():
    Pregnancies = st.sidebar.number_input('Pregnancies', min_value=0, max_value=20, value=1)
    Glucose = st.sidebar.number_input('Glucose', min_value=0, max_value=200, value=85)
    BloodPressure = st.sidebar.number_input('BloodPressure', min_value=0, max_value=140, value=66)
    SkinThickness = st.sidebar.number_input('SkinThickness', min_value=0, max_value=100, value=29)
    Insulin = st.sidebar.number_input('Insulin', min_value=0, max_value=900, value=0)
    BMI = st.sidebar.number_input('BMI', min_value=0.0, max_value=70.0, value=26.6)
    DiabetesPedigreeFunction = st.sidebar.number_input('DiabetesPedigreeFunction', min_value=0.0, max_value=3.0, value=0.351)
    Age = st.sidebar.number_input('Age', min_value=0, max_value=120, value=31)
    
    data = {'Pregnancies': Pregnancies,
            'Glucose': Glucose,
            'BloodPressure': BloodPressure,
            'SkinThickness': SkinThickness,
            'Insulin': Insulin,
            'BMI': BMI,
            'DiabetesPedigreeFunction': DiabetesPedigreeFunction,
            'Age': Age}
    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()

# Combine user input with the entire dataset
# This will be useful for scaling the input
df = pd.concat([input_df, data.drop(columns=['Outcome'])], axis=0)

# Apply scaling
df_scaled = scaler.transform(df)
input_scaled = df_scaled[:1]

# Make prediction
prediction = model.predict(input_scaled)
prediction_proba = model.predict_proba(input_scaled)

# Display user input features
st.subheader('User Input Parameters')
st.write(input_df)

# Display prediction
st.subheader('Prediction')
st.write('**Diabetes Prediction:** ' + ('Positive' if prediction[0] == 1 else 'Negative'))

# Display prediction probability
st.subheader('Prediction Probability')
st.write(prediction_proba)

# Plotting prediction probability
st.subheader('Prediction Probability Chart')
labels = ['Negative', 'Positive']
fig, ax = plt.subplots()
ax.pie(prediction_proba[0], labels=labels, autopct='%1.1f%%', startangle=90)
ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
st.pyplot(fig)

# Add a correlation heatmap to the app
st.subheader('Correlation Heatmap')

# Calculate the correlation matrix
corr = data.corr()

# Create the heatmap
fig, ax = plt.subplots()
sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', ax=ax)
st.pyplot(fig)

# Add a pair plot to the app
st.subheader('Pair Plot')

# Create the pair plot
fig = px.scatter_matrix(data, dimensions=['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'], color='Outcome')
st.plotly_chart(fig)
