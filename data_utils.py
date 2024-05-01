import sklearn 
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
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler


def Time_Formatx(x):
  # Formatting Time
    if pd.isna(x):
        return None
    if x == 2400:
        x = 0
    x = "{0:04d}".format(int(x))
    return datetime.time(int(x[0:2]), int(x[2:4]))

def stats(g):
    # Statistical Information for a Group

    return {'mean':g.mean(), 'variance':g.var(), 'count':g.count(), 'min':g.min(), 'max':g.max()}


Flights = pd.read_csv('flights.csv')
missing_values_flights_csv = Flights.isnull().sum()
print(missing_values_flights_csv)

df_flights_csv = pd.read_csv('flights.csv')
# Determine the size of the dataset
dataset_size = df_flights_csv.shape

# Report the size
print('Dataset size:', dataset_size)

def impute_or_drop(dataframe, columns_to_impute, strategy='mean', drop=False):
    if drop:
        # Drop rows with any null values in specified columns
        dataframe = dataframe.dropna(subset=columns_to_impute)
    else:
        for column in columns_to_impute:
            if dataframe[column].dtype.kind in 'iufc':  # Check if the column is numeric
                if strategy == 'mean':
                    fill_value = dataframe[column].mean()
                elif strategy == 'median':
                    fill_value = dataframe[column].median()
                elif strategy == 'mode':
                    fill_value = dataframe[column].mode()[0]
                dataframe[column].fillna(fill_value, inplace=True)
            else:
                # For non-numeric columns, fill with mode or a placeholder
                fill_value = dataframe[column].mode()[0] if strategy == 'mode' else 'Unknown'
                dataframe[column].fillna(fill_value, inplace=True)
    return dataframe

# In your preprocess function, call impute_or_drop before the analysis
Flights = impute_or_drop(Flights, ['DEPARTURE_DELAY', 'ARRIVAL_DELAY'], strategy='mean', drop=False)