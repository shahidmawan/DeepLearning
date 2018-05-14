import numpy as np
import pandas as pd

from keras.models import Sequential
from keras.layers import Dense,Activation,Layer,Lambda

from sklearn.cross_validation import train_test_split
import keras
import matplotlib.pyplot as plt
from IPython.display import clear_output

class PlotLosses(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.i = 0
        self.x = []
        self.losses = []
        self.val_losses = []
        
        self.fig = plt.figure()
        
        self.logs = []

    def on_epoch_end(self, epoch, logs={}):
        
        self.logs.append(logs)
        self.x.append(self.i)
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))
        self.i += 1
        
        clear_output(wait=True)
        plt.plot(self.x, self.losses, label="loss")
        plt.legend()
        plt.show();
        
plot_losses = PlotLosses()

dataset=pd.read_csv("CancerData.csv")


#print all the entities
print("\n\n")
print(dataset.head())


#Drop raw from Data

dataset=dataset.drop(["id"],axis=1)

print("\n\n")
print(dataset.head())


# Rows and column in the Data
print("\n\n")
print(dataset.shape)


#Check Null Value in the Data
print("\n\n")
print(pd.isnull(dataset).sum())


#Mapping Feature 
FindUnique=dict()
count=0
for i in sorted(dataset["diagnosis"].unique(),reverse=True):
    FindUnique[i]=count
    count=count+1

#Unique Elements of the Feature
print("\n\n")
print(FindUnique)
dataset["diagnosis"]=dataset["diagnosis"].map(FindUnique)


#print Random Sample of the Data
print("\n\n")
print(dataset.sample(5))


#divide dataset into x(input) and y(output)
X=dataset.drop(["diagnosis"],axis=1)
y=dataset["diagnosis"]
print("\n\n")
print(X)
print("\n\n")
print(y)

trainX, testX, trainY, testY = train_test_split(X, y, test_size=0.2, random_state=42)
trainX, valX, trainY, valY = train_test_split(trainX, trainY, test_size=0.2, random_state=42)


arr=[30,50,1]
model=Sequential()
for i in range(len(arr)):
    if i!=0 and i!=len(arr)-1:
        if i==1:
            model.add(Dense(arr[i],input_dim=arr[0],kernel_initializer='normal', activation='relu'))
        else:
            model.add(Dense(arr[i],activation='relu'))
model.add(Dense(arr[-1],kernel_initializer='normal',activation="sigmoid"))
model.compile(loss="binary_crossentropy",optimizer='rmsprop',metrics=['accuracy'])



#train Our Model
model.fit(np.array(trainX),np.array(trainY),epochs=40,callbacks=[plot_losses])


#evaluate Our Model

scores=model.evaluate(np.array(valX),np.array(valY))


print("Loss:",scores[0])
print("Accuracy",scores[1]*100)


predY=model.predict(np.array(testX))
print("\n\n")
print(predY)
predY=np.round(predY).astype(int).reshape(1,-1)[0]
print("\n\n")
print(predY)

from sklearn.metrics import confusion_matrix
m=confusion_matrix(predY,testY)
tn, fn, fp, tp=confusion_matrix(predY,testY).ravel()
m=pd.crosstab(predY,testY)
print("Confusion matrix")
print(m)

#Precision: fraction of retrieved docs that are relevant 
P = tp/(tp + fp)
print("Precision = " + str(P))

#Recall: fraction of relevant docs that are retrieved 
R = tp/(tp + fn)

print("Recall = " + str(R))




