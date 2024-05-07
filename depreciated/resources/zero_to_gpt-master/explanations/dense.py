import os,sys
print(f"\t\t\t\t\<{os.getcwd()}>")

#C:\Users\jump3\Desktop\curr school\ECE1895\ece1895-ML_JuniorDesign\resources\zero_to_gpt-master\explanations\dense.py
import pandas as pd

# Read in the data
data = pd.read_csv("resources/zero_to_gpt-master/data/clean_weather.csv", index_col=0)
# Fill in any missing values in the data with past values
data = data.ffill()

# Create a scatter plot of tmax and tmax_tomorrow
data.plot.scatter("tmax", "tmax_tomorrow")

#------------
import matplotlib.pyplot as plt
data.plot.scatter("tmax", "tmax_tomorrow")

# Calculate the prediction given our weight and bias
prediction = lambda x, w1=.82, b=11.99: x * w1 + b

# Plot a linear regression line over our data
plt.plot([30, 120], [prediction(30),prediction(120)], 'green')

#------------
import numpy as np

def mse(actual, predicted):
    # Calculate mean squared error
    return np.mean((actual - predicted) ** 2)

def pred_print(curr,wt,bi):
    print(f'PREDICTION w/ curr={curr},weight={wt},bias={bi}:\t\t{mse(data["tmax_tomorrow"], prediction(curr, wt, bi))}')


pred_print(data["tmax"], .83, 12)
pred_print(66, .83, 12)
pred_print(68, .83, 12)
pred_print(73, .83, 12)
pred_print(78, .89, 16)
pred_print(45, .75, 1)
pred_print(53, .23, 5)