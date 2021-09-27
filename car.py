import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sbn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#create dataframe
dataFrame = pd.read_excel('merc.xlsx')

#checking for null values
#print(dataFrame.isnull().sum())

#remove highest price cars

#len(dataFrame) * 0.01 =~ 131
dataFrame = dataFrame.sort_values('price', ascending=False).iloc[131:]

#remove year = 1970 cars
dataFrame = dataFrame[dataFrame.year != 1970]

#remove transmission values
dataFrame = dataFrame.drop('transmission', axis = 1)

y = dataFrame['price'].values
x = dataFrame.drop('price', axis= 1).values

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=10)

scaler = MinMaxScaler()

x_train = scaler.fit_transform(x_train)
x_test = scaler.fit_transform(x_test)

model = Sequential()

model.add(Dense(12,activation='relu'))
model.add(Dense(12,activation='relu'))
model.add(Dense(12,activation='relu'))
model.add(Dense(12,activation='relu'))

model.add(Dense(1))

model.compile(optimizer='adam', loss= 'mse')

model.fit(x= x_train, y= y_train,validation_data= (x_test, y_test),batch_size= 250, epochs= 300)

#test case
testSeries = {'year': 2017, 'mileage': 15258, 'tax': 30, 'mpg': 64.2, 'engineSize':2.1}
testSeries = pd.Series(testSeries)

testSeries = scaler.transform(testSeries.values.reshape(-1,5))
print(model.predict(testSeries))
