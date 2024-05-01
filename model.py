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
#from data_analysis import dataOverview
from data_analysis import exploratoryDataAnalysis
from data_analysis import preprocess
from keras.models import load_model

'''
def lstm():
    # Load and preprocess the data
    X_train, y_train, X_val, y_val, X_test, y_test = preprocess()
    X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
    X_val = X_val.reshape(X_val.shape[0], 1, X_val.shape[1])
    X_test = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])
    
    # Build the model
    model = Sequential([
        LSTM(units=256, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
        LSTM(units=128, return_sequences=True),
        Dropout(0.2),
        LSTM(units=64, return_sequences=True),
        Dropout(0.2),
        LSTM(units=32, return_sequences=False),
        Dropout(0.2),
        Dense(units=16, activation='sigmoid'),
        Dropout(0.5),
        Dense(units=1, activation='sigmoid')
    ])

    # Compile the model
    optimizer_ = Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer_, loss='binary_crossentropy', metrics=['accuracy'])

    # Train the model
    history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_val, y_val), verbose=1)

    # Evaluate the model
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=1)
    print(f"Test Accuracy: {test_accuracy:.4f}")

    # Save the model
    model.save('lstm_model.h5')

# Call the lstm function
#lstm()

def load_and_predict():
    loaded_model = load_model('lstm_model.h5')
    X_train, y_train, X_val, y_val, X_test, y_test = preprocess()
    X_test = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])
    predictions = loaded_model.predict(X_test)
    print("final results ====== ", predictions)

# Call the lstm function to train, evaluate, and save the model
#lstm()

# Optionally, call the load_and_predict function to load the saved model and make predictions
load_and_predict()'''

def lstm():
    # Load and preprocess the data
    X_train, y_train, X_val, y_val, X_test, y_test = preprocess()
    X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
    X_val = X_val.reshape(X_val.shape[0], 1, X_val.shape[1])
    X_test = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])
    
    # Build the model
    model = Sequential([
        LSTM(units=256, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
        LSTM(units=128, return_sequences=True),
        Dropout(0.2),
        LSTM(units=64, return_sequences=True),
        Dropout(0.2),
        LSTM(units=32, return_sequences=False),
        Dropout(0.2),
        Dense(units=16, activation='sigmoid'),
        Dropout(0.5),
        Dense(units=1, activation='sigmoid')
    ])

    # Compile the model
    optimizer_ = Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer_, loss='binary_crossentropy', metrics=['accuracy'])

    # Train the model
    history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_val, y_val), verbose=1)

    # Evaluate the model
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=1)
    print(f"Test Accuracy: {test_accuracy:.4f}")

    # Save the model
    model.save('lstm_model.h5')

def load_and_predict():
    # Load the model
    loaded_model = load_model('lstm_model.h5')
    
    # Preprocess the data again or load existing test data
    X_train, y_train, X_val, y_val, X_test, y_test = preprocess()
    X_test = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])
    
    # Make predictions
    predictions = loaded_model.predict(X_test)
    print("Final results ====== ", predictions)

# Call the lstm function to train, evaluate, and save the model
lstm()
load_and_predict()