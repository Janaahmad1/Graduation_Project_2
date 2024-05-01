#import sklearn 
import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
import datetime
import seaborn as sns
import matplotlib.pyplot as plt
from math import sin
from math import cos
from math import pi
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from matplotlib.colors import LinearSegmentedColormap
from keras.models import Sequential
from keras.layers import LSTM, Dense
from tensorflow import keras
from keras.models import Model
from keras.layers import Dense
from keras.layers import Dropout
from keras.optimizers import Adam
from data_utils import stats
from data_utils import Time_Formatx
from data_utils import impute_or_drop
from sklearn.model_selection import train_test_split



def dataOverview(Airlines, Airports, Flights):

    # Dataset descriptions

    print(Flights.info(verbose = True, null_counts=True))
    print(Airlines.info(verbose = True, null_counts=True))
    print(Airports.info(verbose = True, null_counts=True))

#YEAR, MONTH, DAY_OF_WEEK, AIRLINE, ORIGIN_AIRPORT, and DESTINATION_AIRPORT
    # Cancellation Reasons

    cancelled = Flights['CANCELLATION_REASON']
    cancelled.dropna(inplace=True)
    cancelledCount = dict(cancelled.value_counts())
    labels = ['Weather','Airline','National Air System','Security']
    sizes = cancelledCount.values()

    fig, ax = plt.subplots(figsize=(8,8))
    ax.pie(sizes, labels=labels, pctdistance=1.25, labeldistance=1.45, autopct='%1.2f%%', startangle=90, textprops={'fontsize': 20})
    ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    plt.show()


    # Flights on Different Days of Week

    daysOfWeek = Flights['DAY_OF_WEEK']
    dayCounts = dict(daysOfWeek.value_counts())
    dayFreq = {}
    for day in sorted(dayCounts):
        dayFreq[day] = dayCounts[day]

    plt.figure(figsize=(12,8))
    flightFreq = list(dayFreq.values())
    flightFreq.append(dayFreq[1]) # add monday
    flightFreq.append(dayFreq[2]) # add tuesday
    days = ['Mon','Tue','Wed','Thu','Fri','Sat','Sun','Mon.','Tue...']
    plt.plot(days,flightFreq)
    plt.xlabel("Days of week", fontsize=16)
    plt.ylabel("No of flights", fontsize=16)
    plt.title("No of flights on days of week", fontsize=16)
    plt.tick_params(labelsize=16)
    plt.show()


    # Flights in Different Months

    months = Flights['MONTH']
    monthCounts = dict(months.value_counts())
    monthFreq = {}
    for month in sorted(monthCounts):
        monthFreq[month] = monthCounts[month]

    plt.figure(figsize=(12,8))
    flightFreq = list(monthFreq.values())
    monthsArr = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    plt.plot(monthsArr, flightFreq)
    plt.xlabel("Months", fontsize=16)
    plt.ylabel("No of flights", fontsize=16)
    plt.title("No of flights on different months", fontsize=16)
    plt.tick_params(labelsize=16)
    plt.show()


    # Delay Threshold

    ttl = Flights.shape[0]
    threshold = 3
    delayLessThanThreshold = Flights[Flights['ARRIVAL_DELAY'] <= threshold].shape[0] / ttl
    print(delayLessThanThreshold)

def exploratoryDataAnalysis(df):

    # # Overall analysis

    # report = sv.analyze(df)
    # report.show_html("EDA.html")


    # Air Traffic Share of Airlines

     plt.subplots(figsize=(15,20))
     plt.pie(df['AIRLINE'].value_counts(),labels=df['AIRLINE_NAME'].unique(),autopct='%1.0f%%',textprops={'fontsize': 20})
     plt.show()


    # Calculating Data Statistics

     Origin_Stats = df['ARRIVAL_DELAY'].groupby(df['ORIGIN']).apply(stats).unstack().sort_values('count',ascending=False)
     Destination_Stats = df['ARRIVAL_DELAY'].groupby(df['DESTINATION']).apply(stats).unstack().sort_values('count',ascending=False)
     Airline_Stats = df['ARRIVAL_DELAY'].groupby(df['AIRLINE']).apply(stats).unstack().sort_values('mean')
     print(Airline_Stats)


    # Airline Delays on Different Days of Week ********* changes

     Days = ["Monday", "Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
     Airline_Day_Stats = pd.DataFrame()
     for a in df['AIRLINE'].unique():
         x = df[df['AIRLINE']==a]
         t = x['ARRIVAL_DELAY'].groupby(df['DAY']).mean()
         Airline_Day_Stats[a]=t
     Airline_Day_Stats.dropna(inplace=True)
     print(Airline_Day_Stats)

     sns.set(context="paper")
     plt.subplots(figsize=(10,8))
     plt.title("Mean Delay for Airline Vs. Day of Week")
     sns.heatmap(Airline_Day_Stats, linewidths=0.01, cmap=LinearSegmentedColormap.from_list('rg',["g", "w", "r"], N=256),robust=True,yticklabels=Days)
     plt.show()


    # Busiest airports and Airlines

     Airports = df['DESTINATION_CITY'].groupby(df["DESTINATION_CITY"]).count().sort_values(ascending=False).iloc[:11].keys().tolist()
     map = df[['AIRLINE_NAME','DESTINATION_CITY','ARRIVAL_DELAY']]

     frames = list()
     for x in Airports:
         frames.append(map.loc[map["DESTINATION_CITY"] == x])
     map = pd.concat(frames)

     airline_city_delay = pd.DataFrame()
     for airlines in map["AIRLINE_NAME"].unique():
         t = map.loc[map["AIRLINE_NAME"] == airlines]
         temp = t["ARRIVAL_DELAY"].groupby(t["DESTINATION_CITY"]).mean()
         airline_city_delay[airlines] = temp

     sns.set(context="paper")
     plt.subplots(figsize=(10,8))
     plt.title("Mean Delay for Airline Vs. Destination Airports")
     sns.heatmap(airline_city_delay, linewidths=0.01, cmap=LinearSegmentedColormap.from_list('rg',["g", "w", "r"], N=256),robust=True,yticklabels=Airports)
     plt.show()


    # Distance and Delay

     map = df[["DISTANCE","ARRIVAL_DELAY","AIRLINE_NAME"]].copy()
     interval = list()
     for i in range(0,5000,100):
         interval.append(i)

     map["DISTANCE_INTERVAL"] = pd.cut(x = map["DISTANCE"], bins = interval)
     map["DISTANCE_MID"] = map["DISTANCE_INTERVAL"].apply(lambda x : x.mid)
     newMap = map["ARRIVAL_DELAY"].groupby(map["DISTANCE_MID"]).mean().to_frame()
     newMap.dropna(inplace=True)
     newMap.plot.line(title = "Distance vs Delay graph (Bucket Size:100)")
     plt.show()


    # Distribution of Arrival Delay

     sns.displot(df['ARRIVAL_DELAY'], bins = [i for i in range(-50,100)])
     plt.show()



'''
def preprocess(analysis = False):

     Airlines = pd.read_csv('airlines.csv')
     Airports = pd.read_csv('airports.csv')
     Flights = pd.read_csv('flights.csv')

     if analysis:
        dataOverview(Airlines, Airports, Flights)


    # Dropping rows with NaN values and selecting data for January

     Flights = Flights.iloc[:,:23]
     Flights.dropna(inplace=True)
     Flights = Flights[Flights["MONTH"]==1]
     Flights.reset_index(drop=True, inplace=True)


    # Collecting Names of Airlines and Airports
     Airline_Names = {Airlines["IATA_CODE"][i]: Airlines["AIRLINE"][i] for i in range(len(Airlines))}
     Airport_Names = {Airports["IATA_CODE"][i]: Airports["AIRPORT"][i] for i in range(len(Airports))}
     City_Names = {Airports["IATA_CODE"][i]: Airports["CITY"][i] for i in range(len(Airports))}


    # Merging Datasets & Selecting relevant columns

     df = pd.DataFrame()
     df['DATE'] = pd.to_datetime(Flights[['YEAR','MONTH', 'DAY']])
     df['DAY'] = Flights["DAY_OF_WEEK"]
     df['AIRLINE'] = Flights["AIRLINE"]
     df['AIRLINE_NAME'] = [Airline_Names[Flights["AIRLINE"][x]] for x in range(len(Flights))]
     df['FLIGHT_NUMBER'] = Flights['FLIGHT_NUMBER']
     df['TAIL_NUMBER'] = Flights['TAIL_NUMBER']
     df['ORIGIN'] = Flights['ORIGIN_AIRPORT']
     df['ORIGIN_AIRPORT_NAME'] = [Airport_Names[Flights["ORIGIN_AIRPORT"][x]] for x in range(len(Flights))]
     df['ORIGIN_CITY'] = [City_Names[Flights["ORIGIN_AIRPORT"][x]] for x in range(len(Flights))]
     df['DESTINATION'] = Flights['DESTINATION_AIRPORT']
     df['DESTINATION_AIRPORT_NAME'] = [Airport_Names[Flights["DESTINATION_AIRPORT"][x]] for x in range(len(Flights))]
     df['DESTINATION_CITY'] = [City_Names[Flights["DESTINATION_AIRPORT"][x]] for x in range(len(Flights))]
     df['DISTANCE'] = Flights['DISTANCE']
     df['SCHEDULED_DEPARTURE'] = Flights['SCHEDULED_DEPARTURE'].apply(Time_Formatx)
     df['SCHEDULED_ARRIVAL'] = Flights['SCHEDULED_ARRIVAL'].apply(Time_Formatx)
     df['TAXI_OUT'] = Flights['TAXI_OUT']
     df['DEPARTURE_DELAY'] = Flights['DEPARTURE_DELAY']
     df['ARRIVAL_DELAY'] = Flights['ARRIVAL_DELAY']
     df = df[df.ARRIVAL_DELAY < 500]

     if analysis:
       print(df)

     if analysis:
         exploratoryDataAnalysis(df)

    # Selecting Features

     le = LabelEncoder()
     df['TAIL_NUMBER_ENCODED'] = le.fit_transform(df['TAIL_NUMBER'])

    # Handling Date and Time Data

     df['AIRLINE'] = le.fit_transform(df['AIRLINE'])
     df['ORIGIN'] = le.fit_transform(df['ORIGIN'])
     df['DESTINATION'] = le.fit_transform(df['DESTINATION'])

     X = df[['DAY', 'AIRLINE', 'FLIGHT_NUMBER', 'TAIL_NUMBER_ENCODED', 'ORIGIN', 'DESTINATION', 'DISTANCE', 'TAXI_OUT', 'DEPARTURE_DELAY', 'SCHEDULED_DEPARTURE', 'SCHEDULED_ARRIVAL']]
     Y = (df['ARRIVAL_DELAY'] > 15).astype(int)

     Data = Data.drop(['SCHEDULED_DEPARTURE','SCHEDULED_ARRIVAL','DATE'],axis=1)
     Data.dropna(inplace=True)
     Data.reset_index(inplace=True,drop=True)

     if analysis:
         print(Data)


    # Handling Categorical Variables

     L = LabelEncoder()

     Data['AIRLINE']=L.fit_transform(np.array(Data['AIRLINE']).reshape(-1,1))
     Data['ORIGIN']=L.fit_transform(np.array(Data['ORIGIN']).reshape(-1,1))
     Data['DESTINATION']=L.fit_transform(np.array(Data['DESTINATION']).reshape(-1,1))

     H = OneHotEncoder()

     a = pd.DataFrame(H.fit_transform(np.array(Data['AIRLINE']).reshape(-1,1)).toarray())
     a.columns = [str(i) for i in range(len(a.columns))]
     b = pd.DataFrame(H.fit_transform(np.array(Data['ORIGIN']).reshape(-1,1)).toarray())
     b.columns = [str(i+len(a.columns)) for i in range(len(b.columns))]
     c = pd.DataFrame(H.fit_transform(np.array(Data['DESTINATION']).reshape(-1,1)).toarray())
     c.columns = [str(i+len(a.columns)+len(b.columns)) for i in range(len(c.columns))]

     Data = Data.drop(['AIRLINE'],axis=1)
     Data = Data.join(a)
     #Data = Data.drop(['ORIGIN'],axis=1)
     #Data = Data.join(b)
     #Data = Data.drop(['DESTINATION'],axis=1)
     #Data = Data.join(c)
     Data.dropna(inplace=True)

     if analysis:
         print(Data)


    # Splitting into X and Y
    
     X = Data.copy()
     X.drop(['ARRIVAL_DELAY'],axis=1, inplace = True)
     Y = Data['ARRIVAL_DELAY'].copy()

     for i in range(len(Y)):
         if Y[i]<3: Y[i]=0
         else: Y[i]=1

     X = X.to_numpy()
     Y = Y.to_numpy()

     if analysis:
         print("X shape: ", X.shape)
         print("Y shape: ", Y.shape)


    # Splitting into Train, Val, Test

     X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=42, stratify=Y)
     X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state=42, stratify=y_test)

     if analysis:
         print(f"X_train: {X_train.shape}, X_val: {X_val.shape}, X_test: {X_test.shape}")
         print(f"y_train: {y_train.shape}, y_val: {y_val.shape}, y_test: {y_test.shape}")


    # Standard Scaling

     scaler = StandardScaler()
     X_train = scaler.fit_transform(X_train)
     X_val = scaler.transform(X_val)
     X_test = scaler.transform(X_test)
     return X_train, y_train, X_val, y_val, X_test, y_test 
         
'''

def preprocess(analysis=False):
    # Load data
    Airlines = pd.read_csv('airlines.csv')
    Airports = pd.read_csv('airports.csv')
    Flights = pd.read_csv('flights.csv')

    if analysis:
        dataOverview(Airlines, Airports, Flights)  # Assuming dataOverview is a function defined elsewhere

    # Dropping rows with NaN values and selecting data for January
    Flights = Flights.iloc[:,:23]
    Flights.dropna(inplace=True)
    Flights = Flights[Flights["MONTH"] == 1]
    Flights.reset_index(drop=True, inplace=True)

    # Collecting Names of Airlines and Airports
    Airline_Names = {Airlines["IATA_CODE"][i]: Airlines["AIRLINE"][i] for i in range(len(Airlines))}
    Airport_Names = {Airports["IATA_CODE"][i]: Airports["AIRPORT"][i] for i in range(len(Airports))}
    City_Names = {Airports["IATA_CODE"][i]: Airports["CITY"][i] for i in range(len(Airports))}

    # Merging Datasets & Selecting relevant columns
    df = pd.DataFrame()
    df['DATE'] = pd.to_datetime(Flights[['YEAR', 'MONTH', 'DAY']])
    df['DAY'] = Flights["DAY_OF_WEEK"]
    df['AIRLINE'] = Flights["AIRLINE"]
    df['FLIGHT_NUMBER'] = Flights['FLIGHT_NUMBER']
    df['TAIL_NUMBER'] = Flights['TAIL_NUMBER']
    df['ORIGIN'] = Flights['ORIGIN_AIRPORT']
    df['DESTINATION'] = Flights['DESTINATION_AIRPORT']
    df['DISTANCE'] = Flights['DISTANCE']
    df['SCHEDULED_DEPARTURE'] = Flights['SCHEDULED_DEPARTURE'].apply(Time_Formatx)
    df['SCHEDULED_ARRIVAL'] = Flights['SCHEDULED_ARRIVAL'].apply(Time_Formatx)
    df['TAXI_OUT'] = Flights['TAXI_OUT']
    df['DEPARTURE_DELAY'] = Flights['DEPARTURE_DELAY']
    df['ARRIVAL_DELAY'] = Flights['ARRIVAL_DELAY']
    df = df[df.ARRIVAL_DELAY < 500]  # Filter to manage extreme values

    if analysis:
        exploratoryDataAnalysis(df)  # Assuming exploratoryDataAnalysis is a function defined elsewhere

    # Handling Categorical Variables with Label Encoding
    le = LabelEncoder()
    df['AIRLINE'] = le.fit_transform(df['AIRLINE'])
    df['ORIGIN'] = le.fit_transform(df['ORIGIN'])
    df['DESTINATION'] = le.fit_transform(df['DESTINATION'])
    df['TAIL_NUMBER_ENCODED'] = le.fit_transform(df['TAIL_NUMBER'])

    # Selecting Features for Model
    X = df[['DAY', 'AIRLINE', 'FLIGHT_NUMBER', 'TAIL_NUMBER_ENCODED', 'ORIGIN', 'DESTINATION', 'DISTANCE', 'TAXI_OUT', 'DEPARTURE_DELAY']]
    Y = (df['ARRIVAL_DELAY'] > 15).astype(int)  # Binary classification, delay > 15 min

    # Splitting into Train, Validation, and Test sets
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=42, stratify=Y)
    X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state=42, stratify=y_test)

    # Standard Scaling
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    if analysis:
        print(f"X_train: {X_train.shape}, X_val: {X_val.shape}, X_test: {X_test.shape}")
        print(f"y_train: {y_train.shape}, y_val: {y_val.shape}, y_test: {y_test.shape}")

    return X_train, y_train, X_val, y_val, X_test, y_test
