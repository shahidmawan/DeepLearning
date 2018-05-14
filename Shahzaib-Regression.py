import numpy as np
import pandas as pd

from keras.models import Sequential
from keras.layers import Dense,Activation,Layer,Lambda
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.cross_validation import train_test_split


dataset=pd.read_csv("50_Startups.csv")

print(dataset.head())


print(dataset.shape)

print(pd.isnull(dataset).sum())

print("Dataset Features Details")
print(dataset.info())
print("=" * 30)

def mapping(data,feature):
    featureMap=dict()
    count=0
    for i in sorted(data[feature].unique(),reverse=True):
        featureMap[i]=count
        count=count+1
    data[feature]=data[feature].map(featureMap)
    return data



print(dataset["State"].unique())

dataset=mapping(dataset,feature="State")

print(dataset.sample(10))

x=dataset.drop(["Profit"],axis=1)

print(x)

y=dataset["Profit"]
y.head="profit"
y=pd.DataFrame(y)

print(y)

trainX, testX, trainY, testY = train_test_split(x, y, test_size=0.2, random_state=42)
trainX, valX, trainY, valY = train_test_split(trainX, trainY, test_size=0.2, random_state=42)

print(trainX)

print(trainY)

# create model
model = Sequential()
model.add(Dense(4, input_dim=4, kernel_initializer='normal', activation='relu'))
model.add(Dense(1, kernel_initializer='normal'))
# Compile model
model.compile(loss='mean_squared_error', optimizer='adam')


# evaluate model with standardized dataset
model.fit(trainX,trainY, batch_size = 10, epochs = 100)




print(model.predict(x[2:4]))

