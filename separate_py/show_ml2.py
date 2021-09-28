import pickle
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
from scipy import stats
from datetime import datetime
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import math
import re


def prediction_cycling(weight, duration, sports):
    cycling_data = {'weight':[130,130,130,130,130,130,155,155,155,155,155,155,180,180,180,180,180,180,205,205,205,205,205,205],
                'intensity/level':['<10 mph','>20 mph','10-11.9 mph','12-13.9 mph','14-15.9 mph','16-19 mph','<10 mph','>20 mph','10-11.9 mph','12-13.9 mph','14-15.9 mph','16-19 mph','<10 mph','>20 mph','10-11.9 mph','12-13.9 mph','14-15.9 mph','16-19 mph','<10 mph','>20 mph','10-11.9 mph','12-13.9 mph','14-15.9 mph','16-19 mph'], 
                'calories':[236, 944, 354, 472, 590, 708, 281, 1126, 422, 563, 704, 844, 327, 1308, 490, 654, 817, 981, 372, 1489, 558, 745, 931, 1117]}
    
    cycling_df = pd.DataFrame(cycling_data)
    cycling_df['intensity'] = [0 if x == '<10 mph' else 1 if x == '10-11.9 mph' else 2 if x == '12-13.9 mph' else 3 if x == '14-15.9 mph' else 4 if x == '16-19 mph' else 5 for x in cycling_df['intensity/level']]

    cycling_X = cycling_df[["weight","intensity"]]
    cycling_y = cycling_df[["calories"]]
    cycling_X_train,cycling_X_test, cycling_y_train,cycling_y_test = train_test_split(cycling_X,cycling_y,test_size=0.2,random_state=42)
    model1 = LinearRegression()
    model1.fit(cycling_X_train,cycling_y_train)
    cycling_y_pred = model1.predict([[weight, sports]])/60*duration
    return cycling_y_pred

def prediction_running(weight, duration, sports):
    running_data = {'weight':[130,130,130,130,130,130,130,130,130,130,130,155,155,155,155,155,155,155,155,155,155,155,180,180,180,180,180,180,180,180,180,180,180,205,205,205,205,205,205,205,205,205,205,205],
                    'intensity/level': ['5 mph', '5.2 mph', '6 mph', '6.7 mph', '7 mph', '7.5 mph', '8 mph', '8.6 mph', '9 mph', '10 mph', '10.9 mph','5 mph', '5.2 mph', '6 mph', '6.7 mph', '7 mph', '7.5 mph', '8 mph', '8.6 mph', '9 mph', '10 mph', '10.9 mph','5 mph', '5.2 mph', '6 mph', '6.7 mph', '7 mph', '7.5 mph', '8 mph', '8.6 mph', '9 mph', '10 mph', '10.9 mph','5 mph', '5.2 mph', '6 mph', '6.7 mph', '7 mph', '7.5 mph', '8 mph', '8.6 mph', '9 mph', '10 mph', '10.9 mph'],
                    'calories': [472, 531, 590, 649, 679, 738, 797, 826, 885, 944, 1062, 563, 633, 704, 774, 809, 880,950, 985, 1056, 1126, 1267, 654, 735, 817, 899,940, 1022, 1103, 1144, 1226, 1308, 1471, 745, 838, 931, 1024, 1070, 1163, 1256, 1303, 1396, 1489, 1675]}
    
    running_df = pd.DataFrame(running_data)
    running_df['intensity'] = [0 if x == '5 mph' else 1 if x == '5.2 mph' else 2 if x == '6 mph' else 3 if x == '6.7 mph' else 4 if x == '7 mph' else 5 if x == '7.5 mph' else 6 if x == '8 mph' else 7 if x == '8.6 mph' else 8 if x == '9 mph' else 9 if x == '10 mph' else 10 for x in running_df['intensity/level']]
    running_X = running_df[["weight","intensity"]]
    running_y = running_df[["calories"]]
    running_X_train,running_X_test, running_y_train,running_y_test = train_test_split(running_X,running_y,test_size=0.2,random_state=42)
    model2 = LinearRegression()
    model2.fit(running_X_train,running_y_train)
    running_y_pred = model2.predict([[weight, sports]])/60*duration
    return running_y_pred

def prediction_walking(weight, duration, sports):
    walking_data = {'weight':[130,130,130,130,130,130,130,155,155,155,155,155,155,155,180,180,180,180,180,180,180,205,205,205,205,205,205,205],
                    'intensity/level':['2.0 mph', '2.5 mph', '3.0 mph', '3.5 mph', '4.0 mph', '4.5 mph', '5.0 mph','2.0 mph', '2.5 mph', '3.0 mph', '3.5 mph', '4.0 mph', '4.5 mph', '5.0 mph', '2.0 mph', '2.5 mph', '3.0 mph', '3.5 mph', '4.0 mph', '4.5 mph', '5.0 mph', '2.0 mph', '2.5 mph', '3.0 mph', '3.5 mph', '4.0 mph', '4.5 mph', '5.0 mph'],
                    'calories': [148,177,195,224,295,372,472,176,211,232,267,352,443,563,204,245,270,311,409,515,654,233,279,307,354,465,586,745]}

    walking_df = pd.DataFrame(walking_data)
    walking_df['intensity'] = [0 if x == '2.0 mph' else 1 if x == '2.5 mph' else 2 if x == '3.0 mph' else 3 if x == '3.5 mph' else 4 if x == '4.0 mph' else 5 if x == '4.5 mph' else 6 for x in walking_df['intensity/level']]
    walking_X = walking_df[["weight","intensity"]]
    walking_y = walking_df[["calories"]]
    walking_X_train,walking_X_test, walking_y_train,walking_y_test = train_test_split(walking_X,walking_y,test_size=0.2,random_state=42)
    model3 = LinearRegression()
    model3.fit(walking_X_train,walking_y_train)
    walking_y_pred = model3.predict([[weight, sports]])/60*duration
    return walking_y_pred

def prediction_swimming(weight, duration, sports):
    global swimming_df
    swimming_data = {'weight':[130,130,130,130,130,130,130,130,130,130,155,155,155,155,155,155,155,155,155,155,180,180,180,180,180,180,180,180,180,180,205,205,205,205,205,205,205,205,205,205],
                    'intensity/level':['freestyle fast','free style slow','backstroke','breaststroke','butterfly','leisurely','sidestroke','synchronized','trending water fast','trending water moderate','freestyle fast','free style slow','backstroke','breaststroke','butterfly','leisurely','sidestroke','synchronized','trending water fast','trending water moderate','freestyle fast','free style slow','backstroke','breaststroke','butterfly','leisurely','sidestroke','synchronized','trending water fast','trending water moderate','freestyle fast','free style slow','backstroke','breaststroke','butterfly','leisurely','sidestroke','synchronized','trending water fast','trending water moderate'],
                    'calories':[590,413,413,590,649,354,472,472,590,236,704,493,493,704,774,422,563,563,704,281,817,572,572,817,899,490,654,654,817,327,931,651,651,931,1024,558,745,745,931,372]}

    swimming_df = pd.DataFrame(swimming_data)
    swimming_df['intensity'] = [0 if x == 'trending water moderate' else 1 if x == 'leisurely' else 2 if x == 'free style slow' else 3 if x == 'backstroke' else 4 if x == 'sidestroke' else 5 if x == 'synchronized' else 6 if x == 'freestyle fast' else 7 if x == 'breaststroke' else 8 if x == 'trending water fast' else 9 for x in swimming_df['intensity/level']]
    swimming_X = swimming_df[["weight","intensity"]]
    swimming_y = swimming_df[["calories"]]
    swimming_X_train,swimming_X_test, swimming_y_train,swimming_y_test = train_test_split(swimming_X,swimming_y,test_size=0.2,random_state=42)
    model4 = LinearRegression()
    model4.fit(swimming_X_train,swimming_y_train)
    swimming_y_pred = model4.predict([[weight, sports]])/60*duration
    return swimming_y_pred

# st.header('Calories burned calculation')
# st.subheader('Sports Category')

# def app2():
global weight, sports, duration

st.header('Calories burned calculation')
st.subheader('Sports Category')
df = pd.read_csv('/Users/Calvin/Documents/GitHub/yolov5_streamlit/csv files/exercise_dataset_category2.csv')
df.rename(columns={'Activity, Exercise or Sport (1 hour)':'Sports'}, inplace=True)

#Top Sports DataFrame Only
trying = df.loc[df['Category'].str.contains('Cycling|Running|Walking')] #have certain standard
trying2 = df.loc[df['Category'].str.contains('Swimming')] #pose only
trying2 = trying2.sort_values(by='Calories per kg')

#trying is new DataFrame
#category_list = ['None']
category_list = trying['Category'].apply(lambda x: x.lower()).value_counts().sort_index(ascending=True).index.tolist()
category_list.append('swimming')
sports_list = trying['Sports'].apply(lambda x: x.lower()).value_counts().sort_index(ascending=True).index.tolist()
sports_list_swimming = trying2['Sports'].tolist()
options_category = list(range(len(category_list)))
# options_category = options_category.append(3)

#Choice1
category = st.selectbox('Select your exercise category', options_category, format_func=lambda x: category_list[x])

#list in each category
options_cycling = list(range(len(sports_list[0:6]))) #c0
display_cycling = sports_list[0:6]
options_running = list(range(len(sports_list[7:18]))) #c1
display_running = sports_list[7:18]
options_walking = list(range(len(sports_list[22:30]))) #c2
display_walking = sports_list[22:30]
options_swimming = list(range(len(sports_list_swimming[0:11]))) #c3
display_swimming = sports_list_swimming[0:11]
#Choice2 with condition
if category == options_category[0]:
    st.subheader('Intensity Selection')
    sports = st.selectbox('Select your exercise', options_cycling, format_func=lambda x: display_cycling[x])
elif category == options_category[1]:
    st.subheader('Intensity Selection')
    sports = st.selectbox('Select your exercise', options_running, format_func=lambda x: display_running[x])
elif category == options_category[2]:
    st.subheader('Intensity Selection')
    sports = st.selectbox('Select your exercise', options_walking, format_func=lambda x: display_walking[x])
elif category == options_category[3]:
    st.subheader('Intensity Selection')
    sports = st.selectbox('Select your exercise', options_swimming, format_func=lambda x: display_swimming[x])

#current weight
weight = st.number_input('Weight (kg)', step = 1)*2.2
st.write(weight, ' lbs')
#each exercise duration
duration = st.number_input('Sports Duration in each attempt (minutes)', step = 1)
#daily calories burned:
if st.button('Confirm'):
    if category == 0:
        calories = pd.to_numeric(prediction_cycling(weight, duration, sports)[-1,0])
        st.write('In this attempt, you have reduced: ',pd.to_numeric(prediction_cycling(weight, duration, sports)[-1,0]), 'calories in exercise')
        #st.write(calories)
    if category == 1:
        calories = pd.to_numeric(prediction_running(weight, duration, sports)[-1,0])
        st.write('In this attempt, you have reduced: ',prediction_running(weight, duration, sports)[-1,0], 'calories in exercise')
    if category == 2:
        calories = pd.to_numeric(prediction_walking(weight, duration, sports)[-1,0])
        st.write('In this attempt, you have reduced: ',prediction_walking(weight, duration, sports)[-1,0], 'calories in exercise')
    if category == 3:
        calories = pd.to_numeric(prediction_swimming(weight, duration, sports)[-1,0])
        st.write('In this attempt, you have reduced: ',prediction_swimming(weight, duration, sports)[-1,0], 'calories in exercise')
#1 global variable from function in the same file
#2 prediction model


if __name__ == '__app__':
    app2()
