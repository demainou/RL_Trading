# Import the numpy and math modules, as follows:
import numpy as np
import math

# Next, define a function to format the price to two decimal places, to reduce the ambiguity of the data:
def formatPrice(n):

    if n>=0:
        curr = "$"
    else:
        curr = "-$"
    return (curr +"{0:.2f}".format(abs(n)))

# Return a vector of stock data from the CSV file. Convert the closing stock prices from the data to vectors, and return a vector of all stock prices, as follows:
def getStockData(key):
    datavec = []
    lines = open("data/" + key + ".csv", "r").read().splitlines()

    for line in lines[1:]:
        datavec.append(float(line.split(",")[4]))

    return datavec

# Next, define a function to generate states from the input vector. Create the time series by generating
# the states from the vectors created in the previous step. The function for this takes three parameters:
#  the data; a time, t (the day that you want to predict); and a window (how many days to go back in time).
# The rate of change between these vectors will then be measured and based on the sigmoid function:
def getState(data, t, window):
    if t - window >= -1:
        vec = data[t - window+ 1:t+ 1]
    else:
        vec = -(t-window+1)*[data[0]]+data[0: t + 1]
    scaled_state = []
    for i in range(window - 1):

# Next, scale the state vector from 0 to 1 with a sigmoid function. The sigmoid function can map any
# input value, from 0 to 1. This helps to normalize the values to probabilities:
        scaled_state.append(1/(1 + math.exp(vec[i] - vec[i+1])))
    return np.array([scaled_state])

