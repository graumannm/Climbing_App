import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# run code to fit model on existing data
# load data 
climbing = pd.read_csv("./climbing_dataset/climber_df.csv")

# which target we will try to predict
target = 'grades_max'

# first drop country and date first and last and predict grades max
X = climbing.drop(['user_id','grades_count','grades_first','grades_last','grades_max','grades_mean','date_first','date_last'],axis=1)

# one hot encode country
one_hot_encoded = pd.get_dummies(X['country'])

# Concatenate the one-hot encoded columns with the original DataFrame
X_encoded = pd.concat([X, one_hot_encoded], axis=1)

# remove original country column
X_encoded.drop(labels='country',axis=1, inplace=True)

# feature engineering: compute BMI
X_encoded['BMI'] = X_encoded['weight'] / (X_encoded['height']/100)**2

# define y: highest grade climbed
y = climbing[target]

# make predictions
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

# Create a Linear Regression model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# evaluate predictions
from sklearn.metrics import mean_squared_error
import numpy as np

# Calculate mean squared error
mse = mean_squared_error(y_test, y_pred)

# Calculate root mean squared error
rmse = np.sqrt(mse)

# make a prediction for a new row/person
if __name__ == '__main__':
    st.title('Climbing grade prediction')

sex = st.slider('Gender. 0=male, 1=femaile',min_value=0, max_value=1)
height = st.slider('Height in cm',min_value=0, max_value=210)
weight = st.slider('Weight in kg',min_value=0, max_value=200)
age = st.slider('Age',min_value=0, max_value=100)
years_cl = st.slider('How many years have you been climbing?',min_value=0, max_value=60)
year_first = st.slider('Which year did you start climbing?',min_value=1950, max_value=2023) 
year_last = st.slider('Which year was the last time you went climbing?',min_value=1950, max_value=2023) 
country = st.selectbox('Which country are you from?',
                       ['AUS', 'AUT', 'BEL', 'BRA', 'CAN', 'CHE', 'CZE', 'DEU', 'DNK', 'ESP',
                        'FIN', 'FRA', 'GBR', 'HRV', 'ITA', 'MEX', 'NLD', 'NOR', 'POL', 'PRT',
                        'RUS', 'SVN', 'SWE', 'USA', 'ZAF', 'other']) 

new_climber = pd.DataFrame({
                        'sex': sex, 
                        'height': height, 
                        'weight': weight, 
                        'age': age, 
                        'years_cl': years_cl,
                        'year_first': year_first, 
                        'year_last': year_last, 
                        'country': country},
                        index=[0])

# create 1-hot encoding for country 

# make dictionary of 0's for each country
encoded_dict = dict(zip(climbing['country'].unique(), np.zeros(26)))

# insert 1 at new climber's country
encoded_dict[new_climber['country'][0]] = 1

# turn into df
new_encoded = pd.DataFrame(encoded_dict, index=[0])

# add new columns to new climber's DF
new_climber = pd.concat([new_climber, new_encoded], axis=1)

# remove country again
new_climber.drop('country',axis=1, inplace=True)

# calculate BMI for new climber
new_climber['BMI'] = new_climber['weight'] / (new_climber['height']/100)**2

if st.button('Predict climbing grade'):
    # Make prediction for the new row
    prediction = model.predict(new_climber)

    # convert to French scale and output result
    conversion_table = pd.read_csv("./climbing_dataset/grades_conversion_table.csv")
    my_grade  = int(np.rint(prediction))
    row_index = conversion_table.loc[conversion_table['grade_id'] == my_grade].index[0]
    french_grade = conversion_table.loc[row_index,'grade_fra']
    prediction_string = f'Your predicted climbing grade is {french_grade}'
    st.text(f'Your prediction: {prediction_string}')