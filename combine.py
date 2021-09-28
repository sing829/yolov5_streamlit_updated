import streamlit as st
import pandas as pd
#############################
from io import StringIO
from pathlib import Path
import streamlit as st
import time
from detect import detect
import os
import sys
import argparse
from PIL import Image
import pandas as pd
import re
import numpy as np
#######################
import pickle
######################
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
from streamlit.elements.button import FORM_DOCS_INFO
import json
import SessionState
import webbrowser

#file route: Line 68, 93, 95, 214, 307, 400, 401
# Security
#passlib,hashlib,bcrypt,scrypt
import hashlib
def make_hashes(password):
    return hashlib.sha256(str.encode(password)).hexdigest()

def check_hashes(password,hashed_text):
    if make_hashes(password) == hashed_text:
        return hashed_text
    return False

###############################################################################################################
# DB Management
import sqlite3 
conn = sqlite3.connect('data.db')
c = conn.cursor()
# DB  Functions
def create_usertable():
    c.execute('CREATE TABLE IF NOT EXISTS userstable(username TEXT,password TEXT)')

def add_userdata(username,password):
    c.execute('INSERT INTO userstable(username,password) VALUES (?,?)',(username,password))
    conn.commit()

def login_user(username,password):
    c.execute('SELECT * FROM userstable WHERE username =? AND password = ?',(username,password))
    data = c.fetchall()
    return data

def view_all_users():
    c.execute('SELECT * FROM userstable')
    data = c.fetchall()
    return data

############################################################################################################

def main():
    display = Image.open('photos/logo.png')
    display = np.array(display)
    # st.image(display, width = 400)
    # st.title("Data Storyteller Application")
    
    st.image(display, width = 320)

    # col
    # """Login"""

    # st.title("Login")

    menu = ["Welcome","SignUp", "Login", "Sports Calories Prediction", "Food Detection", "Suppliment Recommendation","Recipe Recommendation"]
    choice = st.sidebar.selectbox("Menu",menu)

#######################

    if choice == "Welcome":
        st.header("Home Page")
        st.subheader("Welcome to YOO Health Fitness. We are here to guide you through your fitness journey. Are you ready to begin a healthy lifestyle?")


        # col1, col2 = st.beta_columns(2)
        # col2 = st.image(clap_hand, width = 250)

        fitness = Image.open('photos/fiteness_man.jpeg')
        fitness = np.array(fitness)
        clap_hand = Image.open('photos/clap.jpeg')
        clap_hand = np.array(clap_hand)

        st.image(fitness, width = 400)

        st.subheader("Set your goal and follow our instruction to reach your goal!")
        st.image(clap_hand, width = 400)

#######################

    elif choice == "Login":
        st.header("Login Section")

        username = st.text_input("Username")
        password = st.text_input("Password",type='password')
        if st.button("Login"):
            # if password == '12345':
            create_usertable()
            hashed_pswd = make_hashes(password)

            result = login_user(username,check_hashes(password,hashed_pswd))
            if result:

                st.success("Logged In as {}".format(username))

                #task = st.selectbox("Task",["Personal information","Analytics"])
                #if task == "Personal information":
                #    st.subheader("Information")
#
                #elif task == "Analytics":
                #    st.subheader("Analytics")
                
            else:
                st.warning("Incorrect Username/Password")

#######################

    elif choice == "SignUp":
        st.header("Create New Account")
        new_user = st.text_input("Username")
        new_password = st.text_input("Password",type='password')

        if st.button("SignUp"):
            create_usertable()
            add_userdata(new_user,make_hashes(new_password))
            st.success("You have successfully created a valid Account")
            st.info("Please go to Login page to login")

#######################

    elif choice == "Sports Calories Prediction":

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

        # def app2():
        global weight, sports, duration

        st.title('Calories burned calculation')
        st.header('Personal information')
        
        #gender
        CHOICES_GENDER = {1:'Male', 2:'Female'}
        display_gender = ("Male", "Female")
        options_gender = list(range(len(display_gender)))
        Gender = st.selectbox('Gender', options_gender, format_func=lambda x: display_gender[x])

        #age
        Age = st.slider('Age', step = 1, max_value=100, min_value=0)
        #height
        height = st.number_input('Height (cm)', step = 1, min_value=0)
        #weight
        weight = st.number_input('Weight (kg)', step = 1, min_value=0)*2.2
        st.write(weight, ' lbs')
        #expected weight
        expected_weight = st.number_input('Expected Weight (kg)', step = 1, min_value=0)*2.2
        st.write(expected_weight, ' lbs')

        st.header('Sports Category')
        df = pd.read_csv('csv_files/exercise_dataset_category2.csv')
        df.rename(columns={'Activity, Exercise or Sport (1 hour)':'Sports'}, inplace=True)
        #Top Sports DataFrame Only
        trying = df.loc[df['Category'].str.contains('Cycling|Running|Walking')] #have certain standard
        trying2 = df.loc[df['Category'].str.contains('Swimming')] #pose only
        trying2 = trying2.sort_values(by='Calories per kg')
        #trying is new DataFrame
        
        category_list = trying['Category'].apply(lambda x: x.lower()).value_counts().sort_index(ascending=True).index.tolist()
        category_list.append('swimming')
        sports_list = trying['Sports'].apply(lambda x: x.lower()).value_counts().sort_index(ascending=True).index.tolist()
        sports_list_swimming = trying2['Sports'].tolist()
        options_category = list(range(len(category_list)))
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
        

        #each exercise duration
        duration = st.number_input('Sports Duration in each attempt (minutes)', step = 1, min_value=0)
        
        prediction_list = []
        BMR_list_male = []
        BMR_list_female = []
        
        #daily calories burned:
        clicktimes = SessionState.get(x=0)
        if st.button('Confirm'):
            clicktimes.x = clicktimes.x + 1
            if category == 0:
                calories = [pd.to_numeric(prediction_cycling(weight, duration, sports)[-1,0])]
                st.write('In this attempt, you have reduced: ',pd.to_numeric(prediction_cycling(weight, duration, sports)[-1,0]), 'calories in exercise')
                prediction_list.append(pd.to_numeric(prediction_cycling(weight, duration, sports)[-1,0]))

            if category == 1:
                calories = [pd.to_numeric(prediction_running(weight, duration, sports)[-1,0])]
                st.write('In this attempt, you have reduced: ',pd.to_numeric(prediction_running(weight, duration, sports)[-1,0]), 'calories in exercise')
                prediction_list.append(pd.to_numeric(prediction_running(weight, duration, sports)[-1,0]))
                
            if category == 2:
                calories = [pd.to_numeric(prediction_walking(weight, duration, sports)[-1,0])]
                st.write('In this attempt, you have reduced: ', pd.to_numeric(prediction_walking(weight, duration, sports)[-1,0]), 'calories in exercise')
                prediction_list.append(pd.to_numeric(prediction_walking(weight, duration, sports)[-1,0]))
                
            if category == 3:
                calories = [pd.to_numeric(prediction_swimming(weight, duration, sports)[-1,0])]
                st.write('In this attempt, you have reduced: ', pd.to_numeric(prediction_swimming(weight, duration, sports)[-1,0]), 'calories in exercise')
                prediction_list.append(pd.to_numeric(prediction_swimming(weight, duration, sports)[-1,0]))

        #to verify
        #st.write(prediction_list)
        st.subheader('Exercising Calories Table')
        
        attempt_list = []
        lbs = np.array(prediction_list) / 3500
        kg = np.array(prediction_list) / 7700
        #for i in range(1, clicktimes.x):
        for i in range(clicktimes.x):
            attempt_list.append(i)

        record_data = {'Attempt': [clicktimes.x],
                    'Calories': prediction_list,
                    'lbs': lbs,
                    'kg': kg}
        
        record = pd.DataFrame.from_dict(record_data, orient='index')
        record = record.transpose()
        
        st.table(record)

        st.header('Daily Calories Buring')
        st.write('Estimated Daily calories burning without exercising:')
        st.write('The Basal Metabolic Rate (BMR) Calculator estimates your basal metabolic rate—the amount of energy expended while at rest in a neutrally temperate environment. In our calculation we use Revised Harris-Benedict Equation')
        if st.button('Find out more about BMR'):
            webbrowser.open_new_tab("https://en.wikipedia.org/wiki/Basal_metabolic_rate")

        global BMR_Women, BMR_Men
        if Gender == 0:
            BMR_Men = 13.397*(weight/2.2)+4.799*height-5.677*Age+88.362
            st.write('Your BMR in calories/day: ', BMR_Men)
            st.write('in lbs: ', BMR_Men / 3500)
            st.write('in kgs: ', BMR_Men / 7700)
            BMR_list_male.append(BMR_Men)
        if Gender == 1:
            BMR_Women = 9.247*(weight/2.2)+3.098*height-4.330*Age+447.593
            st.write('Your BMR in calories/day: ', BMR_Women)
            st.write('in lbs: ', BMR_Women / 3500)
            st.write('in kgs: ', BMR_Women / 7700)
            BMR_list_male.append(BMR_Women)

        st.subheader('Distance to your target weight')
        if Gender == 0:
            st.write('Remaining Target (calories):', ((weight/2.2)*7700 - (expected_weight/2.2)*7700 - BMR_Men - prediction_list[0]))
            st.write('Remaining Target (lbs):', ((weight/2.2)*7700 - (expected_weight/2.2)*7700 - BMR_Men- prediction_list[0])/3500)
            st.write('Remaining Target (kg):', ((weight/2.2)*7700 - (expected_weight/2.2)*7700 - BMR_Men- prediction_list[0])/7700)
        if Gender == 1:
            st.write('Remaining Target (calories):', ((weight/2.2)*7700 - (expected_weight/2.2)*7700- BMR_Women- prediction_list[0]))   
            st.write('Remaining Target (lbs):', ((weight/2.2)*7700 - (expected_weight/2.2)*7700 - BMR_Women- prediction_list[0])/3500)
            st.write('Remaining Target (kg):', ((weight/2.2)*7700 - (expected_weight/2.2)*7700 - BMR_Women- prediction_list[0])/7700)

#######################

    elif choice == 'Food Detection':

        def get_subdirs(b='.'):
            '''
                Returns all sub-directories in a specific Path
            '''
            result = []
            for d in os.listdir(b):
                bd = os.path.join(b, d)
                if os.path.isdir(bd):
                    result.append(bd)
            return result


        def get_detection_folder():
            '''
                Returns the latest folder in a runs\detect
            '''
            return max(get_subdirs(os.path.join('runs', 'detect')), key=os.path.getmtime)


        if __name__ == '__main__':
            
            st.header('Food Image Detection')

            parser = argparse.ArgumentParser()
            parser.add_argument('--weights', nargs='+', type=str,
                                default='weights/best.pt', help='model.pt path(s)')
            parser.add_argument('--source', type=str,
                                default='data/images', help='source')
            parser.add_argument('--img-size', type=int, default=640,
                                help='inference size (pixels)')
            parser.add_argument('--conf-thres', type=float,
                                default=0.50, help='object confidence threshold')
            parser.add_argument('--iou-thres', type=float,
                                default=0.45, help='IOU threshold for NMS')
            parser.add_argument('--device', default='',
                                help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
            parser.add_argument('--view-img', action='store_true',
                                help='display results')
            parser.add_argument('--save-txt', action='store_true',
                                help='save results to *.txt')
            parser.add_argument('--save-conf', action='store_true',
                                help='save confidences in --save-txt labels')
            parser.add_argument('--nosave', action='store_true',
                                help='do not save images/videos')
            parser.add_argument('--classes', nargs='+', type=int,
                                help='filter by class: --class 0, or --class 0 2 3')
            parser.add_argument('--agnostic-nms', action='store_true',
                                help='class-agnostic NMS')
            parser.add_argument('--augment', action='store_true',
                                help='augmented inference')
            parser.add_argument('--update', action='store_true',
                                help='update all models')
            parser.add_argument('--project', default='runs/detect',
                                help='save results to project/name')
            parser.add_argument('--name', default='exp',
                                help='save results to project/name')
            parser.add_argument('--exist-ok', action='store_true',
                                help='existing project/name ok, do not increment')
            opt = parser.parse_args()
            print(opt)

            source = ("Image Detection", "Video Detection")
            source_index = st.sidebar.selectbox("Select Input", range(
                len(source)), format_func=lambda x: source[x])

            if source_index == 0:
                uploaded_file = st.sidebar.file_uploader(
                    "Upload Images", type=['png', 'jpeg', 'jpg'])
                if uploaded_file is not None:
                    is_valid = True
                    with st.spinner(text='Loading...'):
                        st.sidebar.image(uploaded_file)
                        picture = Image.open(uploaded_file)
                        picture = picture.save(f'data/images/{uploaded_file.name}')
                        opt.source = f'data/images/{uploaded_file.name}'
                else:
                    is_valid = False
            else:
                uploaded_file = st.sidebar.file_uploader("Upload Video", type=['mp4'])
                if uploaded_file is not None:
                    is_valid = True
                    with st.spinner(text='Loading...'):
                        st.sidebar.video(uploaded_file)
                        with open(os.path.join("data", "videos", uploaded_file.name), "wb") as f:
                            f.write(uploaded_file.getbuffer())
                        opt.source = f'data/videos/{uploaded_file.name}'
                else:
                    is_valid = False

            if is_valid:
                print('valid')
                st.markdown("Please press the start detection button to start detecting the image")
                if st.button('Start Detection'):

                    detect(opt)
                    
                    if source_index == 0:
                        with st.spinner(text='Preparing Images'):
                            for img in os.listdir(get_detection_folder()):
                                st.image(str(Path(f'{get_detection_folder()}' ) / img))

                            st.balloons()

                            # Import calories csv
                            df = pd.read_csv('csv_files/Calories per 100 grams of food.csv')
                            # Assign food to designated csv row and columns
                            avo = df.loc[df['Name'].str.contains('AVOCADO')].reset_index(drop=True)
                            broc = df.loc[df['Name'].str.contains('BROCCOLI')].reset_index(drop=True)
                            chic = df.loc[df['Name'].str.contains('CHICKEN')].reset_index(drop=True)
                            egg = df.loc[df['Name'].str.contains('EGG')].reset_index(drop=True)
                            rice = df.loc[df['Name'].str.contains('RICE')].reset_index(drop=True)
                            shr = df.loc[df['Name'].str.contains('SHRIMP')].reset_index(drop=True)
                            steak = df.loc[df['Name'].str.contains('STEAK')].reset_index(drop=True)
                            tmt = df.loc[df['Name'].str.contains('TOMATO')].reset_index(drop=True)
                            yogurt = df.loc[df['Name'].str.contains('YOGURT')].reset_index(drop=True)
                            # Food related calories per 100g
                            avo_option = avo.iloc[0,3]
                            broc_option = broc.iloc[2,3]
                            chic_option = chic.iloc[12,3]
                            egg_option = egg.iloc[5,3]
                            rice_option = rice.iloc[3,3]
                            shr_option = shr.iloc[0,3]
                            steak_option = steak.iloc[0,3]
                            tmt_option = tmt.iloc[2,3]
                            yo_option = yogurt.iloc[1,3]
                            food_calories_list = 0

                            for name in detect(opt): 
                                if name == 'AVOCADO' or name == 'AVOCADOS':
                                    st.write(f'Avocado: {avo_option} calories per 100g')
                                    food_calories_list += int(avo_option)
                                elif name == 'BROCCOLI' or name == 'BROCCOLIS':
                                    st.write(f'Broccoli: {broc_option} calories per 100g')  
                                    food_calories_list += int(broc_option) 
                                elif name == 'CHICKEN'or name == 'CHICKENS':
                                    st.write(f'Chicken: {chic_option} calories per 100g')
                                    food_calories_list += int(chic_option) 
                                elif name == 'EGG' or name == 'EGGS':
                                    st.write(f'Egg: {egg_option} calories per 100g')
                                    food_calories_list += int(egg_option) 
                                elif name == 'RICE':
                                    st.write(f'Rice: {rice_option} calories per 100g')
                                    food_calories_list += int(rice_option) 
                                elif name == 'SHRIMP' or name == 'SHRIMPS':
                                    st.write(f'Shrimp: {shr_option} calories per 100g')
                                    food_calories_list += int(shr_option) 
                                elif name == 'STEAK' or name == 'STEAKS':
                                    st.write(f'Steak: {steak_option} calories per 100g')
                                    food_calories_list += int(steak_option) 
                                elif name == 'TOMATO' or name == 'TOMATOS':
                                    st.write(f'Tomato: {tmt_option} calories per 100g')
                                    food_calories_list += int(tmt_option)
                                elif name == 'YOGURT' or name == 'YOGURTS':
                                    st.write(f'Yogurt: {yo_option} calories per 100g')
                                    food_calories_list += int(yo_option) 
                            # st.subheader(detect(opt))
                            
                            st.write(f"You have obtained {food_calories_list} calories in this meal.")                        
                        
                    else:
                        with st.spinner(text='Preparing Video'):
                            for vid in os.listdir(get_detection_folder()):
                                st.video(str(Path(f'{get_detection_folder()}') / vid))

                            st.balloons()

#########################

    elif choice == 'Suppliment Recommendation':

        def recommender(brand_name,keywords=None,price_min=0,price_max=1000000,flavors=None):

            dataset = []
            df = pd.read_csv('csv_files/combine_token_for_app.csv')
            use_cos = pickle.load(open('models/supp_cos_similarity.pkl','rb'))
            if keywords:
                keywords = keywords.split(';')
                keywords = [ky.strip() for ky in keywords]
            try:
                makeup_id = df[df['brand_name']==brand_name].index.values[0]
            except :
                return dataset
            scores = list(enumerate(use_cos[makeup_id]))
            sorted_scores = sorted(scores, key = lambda x: x[1], reverse=True)

            items = [item[0] for item in sorted_scores]
            df = df.iloc[items]
            df["number_of_flavors"].replace({"Unavailable": 1}, inplace=True)
            df["number_of_flavors"] = pd.to_numeric(df["number_of_flavors"])
            if keywords != None :
                df = df[(df['product_description'].str.contains('|'.join(keywords))) & (df['price']>=price_min) & (df['price']<=price_max)]
            else:
                df = df[(df['price']>=price_min) & (df['price']<=price_max) ]
            df.drop_duplicates(subset=['id'], keep='first',inplace=True)
            df = df.reset_index(drop=True)
            if flavors != None :
                for i in range(df.shape[0]) :
                    if df['top_flavor_rated'][i].strip() not in flavors :
                        df = df.drop([i])
                    else:
                        dataset.append({'Brand':df['brand_name'][i],'price':df['price'][i],'flavor':df['top_flavor_rated'][i]})
            return dataset


        st.header("Suppliment Recommandation")

        def get_brand_names():
            df = pd.read_csv('csv_files/combine_token_for_app.csv')
            df = df.drop_duplicates(subset=['brand_name'], keep='first')
            return list(df['brand_name'])

        with st.form(key = "form1"):
            # brand_name = st.text_input(label = "Enter the product brand")
            brand_name = st.selectbox(label = "Select the brand",options=get_brand_names())
            keywords = st.text_input(label = "Enter the keywords (They should be separated by ';' | Example : keyword1;keyword2;keyword3 )")
            price = st.slider("Enter your budget", 1, 1000,(1, 1000))
            flavors_options_single = st.multiselect(
                    'Enter single flavorss',
                    ['unflavored', 'strawberry', 'lemonade', 'cookie', 'pineapple', 'grape', 'raspberry', 'mint', 'pina colada', 'blueberry', 'cherry', 'candy', 'gingerbread', 'chocolate', 'peanut butter', 'maple waffle', 'fruit', 'orange', 'lemon', 'mango', 'peach', 'watermelon', 'coconut', 'vanilla', 'banana', 'apple', 'caramel', 'hazelnut', 'margarita', 'cinnamon', 'coffee', 'buttermilk', 'kiwi', 'dragon fruit', 'brownie', 'rocky road'])

            flavors_options_mix = st.multiselect(
            'Enter mix flavors',
            ['lemonade + raspberry', 'lemonade + blueberry', 'lemonade + strawberry', 'chocolate + mint', 'chocolate + peanut butter', 'chocolate + coconut', 'chocolate + hazelnut', 'mango + peach', 'mango + lemon', 'mango + orange', 'mango + pineapple', 'banana + peanut butter', 'cherry + watermelon', 'candy + watermelon', 'cookie + peanut butter', 'coffee + caramel', 'apple + cinnamon', 'strawberry + pina colada', 'vanilla + caramel'])
            submit = st.form_submit_button(label = "Submit")
        dataset = []
        if submit :
            if keywords.replace('Example : keyword1;keyword2;keyword3',"").strip() != "" :
                dataset = recommender(brand_name.strip(),keywords,int(price[0]),int(price[1]),flavors_options_single+flavors_options_mix)
            else:

                dataset = recommender(brand_name.strip(),price_min=int(price[0]),price_max=int(price[1]),flavors=flavors_options_single+flavors_options_mix)
            if len(dataset) >10 :
                df = pd.DataFrame(dataset[:10])
                st.table(df)
            elif len(dataset) == 0:
                st.write("No results found")

            else:
                df = pd.DataFrame(dataset)
                st.table(df)      
            submit = False
#########################

    elif choice == 'Recipe Recommendation':
        recipe_use_cos = pickle.load(open('models/recipe_cos_similarity.pkl','rb'))
        recipe_df=pd.read_csv('csv_files/recipe_full_df_combine_token.csv')
        
        st.title("Recipe Recommendation")

        def keywords(name): 
            try:
                keywords_product = recipe_df[recipe_df['title'].str.contains(name, case=False)]['title'].head(30)
            except:
                try: 
                    keywords_product  = recipe_df[recipe_df['ingredients'].str.contains(name, case=False)]['title'].head(30)
                except:
                    try:
                        keywords_product  = recipe_df[recipe_df['recipe'].str.contains(name, case=False)]['title'].head(30)
                    except:
                        return 'Keywords does not match'
            return keywords_product.values.tolist()

        def filtered_recipe(food,course_type,meal_type,nutrition_control,diseases_prevention):
            filtered_recipe_pos =  [recipe_df[recipe_df['title']== i].index[0] for i in food]
            recipe_df2 = recipe_df.iloc[filtered_recipe_pos]
            selection = recipe_df2[(recipe_df2['course_type'].str.contains(course_type))|(recipe_df2['meal_type']==meal_type)|(recipe_df2['nutrition_control'].str.contains(nutrition_control))|(recipe_df2['Diseases_prevention'].str.contains(diseases_prevention))]
            return selection['title'].tolist()


        def search_recommend(name): 
            recommend_list = []
            makeup_id = recipe_df[recipe_df['title']==name].index.values[0]
            scores = list(enumerate(recipe_use_cos[makeup_id]))
            sorted_scores = sorted(scores, key = lambda x: x[1], reverse=True)
            sorted_scores = sorted_scores[0:11]
            
            j = 0
            print('The 10 most recommened products: ')
            for item in sorted_scores:
                product_name = recipe_df.iloc[item[0],:]['title']
                recommend_list.append(product_name)
                j = j+1
                if j > 9:
                    break
            return recommend_list

        # if 'selection' not in st.session_state():
        #     st.session_state['selection'] = 'Nothing'
        # if st.button('show'):
        #     st.session_state['selection'] = selection
        #     st.write('You selected:', selection)

        with st.form(key = "form1"):
            name = st.text_input(label = "Enter the keywords of food you want to eat")
            course_type = st.selectbox('Please select preferred course Type', ['All','lunch_dinner','Appertizer_snacks','breakfast_brunch','Desserts','Beverages'])
            meal_type = st.selectbox('Please select preferred meal Type', ['vegetarian','vegan','non-vegetarian','mediterranean_diet'])
            nutrition_control = st.selectbox('Please select preferred meal for nutrition control', ['low_carbohydrate','low_sodium','high_fibre','high_protein','low_fat','low_sodium'])
            diseases_prevention = st.selectbox('Please select preferred meal for diseases prevention',['Diabetes','High_blood_pressure','High_cholesterol','Heartburn','Celiac_disease','Cancer_protection','Inflammatory_conditions','Alzheimer_prevention','IBS'])
            food = keywords(name)
            selection_list = filtered_recipe(food,course_type,meal_type,nutrition_control,diseases_prevention)
            submit = st.form_submit_button(label = "Submit this form")
            selection = st.selectbox('Please',selection_list)
            recommend = st.form_submit_button(label = "recommend")
        # if submit:
            
        # if recommend:
        #     a = search_recommend(selection)
        #     st.write(a)

        if 'selection' not in st.session_state:
            st.session_state['selection'] = 'None'

        if submit:
            st.session_state['selection'] = selection

        NUTRITION_TEMPLATE ="""
        <div>
        <p>Calories: {}</p>
        <p>Total_fat: {}</p>
        <p>Saturated_fat: {}</p>
        <p>Cholesterol: {}</p>
        <p>Sodium: {}</p>
        <p>Carbohydrate: {}</p>
        <p>Dietry_fibre: {}</p>
        <p>Sugar: {}</p>
        <p>Protein: {}</p>
        <p>Vitamin: {}</p>
        <p>Calcium: {}</p>
        <p>Iron: {}</p>
        <p>Potassium: {}</p>
        </div>
        """
        if recommend:
            recommend_list = search_recommend(selection)
            st.write('The 11 most recommend products:')
            for i in recommend_list:
                recipe = recipe_df[recipe_df['title']==i]['recipe'].values[0]
                nutrition = recipe_df[recipe_df['title']==i]['nutrition'].values[0].replace('\'','\"')
                ingredients = recipe_df[recipe_df['title']==i]['ingredients'].values[0]
                nutrition_dict = json.loads(nutrition)
                calories = nutrition_dict['calories']
                total_fat = nutrition_dict['total_fat']
                saturated_fat = nutrition_dict['saturated_fat']
                cholesterol = nutrition_dict['cholesterol']
                sodium = nutrition_dict['sodium']
                carbohydrate = nutrition_dict['carbohydrate']
                dietry_fibre = nutrition_dict['dietry_fibre']
                sugar = nutrition_dict['sugar']
                protein = nutrition_dict['protein']
                vitamin = nutrition_dict['vitamin']
                calcium = nutrition_dict['calcium']
                iron = nutrition_dict['iron']
                potassium = nutrition_dict['potassium']

                st.session_state['selection'] = recommend_list
                st.write(i)
                with st.expander('Ingredients'):
                    st.write(ingredients)
                with st.expander('Recipe'):
                    st.write(recipe)
                with st.expander('Nutrition Tag'):
                    st.write(NUTRITION_TEMPLATE.format(calories,total_fat,saturated_fat,cholesterol,sodium,carbohydrate,dietry_fibre,sugar,protein,vitamin,calcium,iron,potassium),unsafe_allow_html=True,scrolling=True)
        
if __name__ == '__main__':
	main()
