from pandas import read_csv

import matplotlib.pyplot as plt

df=read_csv("log1_keras.csv",usecols=['epoch','acc','loss'])

epoch=list(df['epoch'])
acc=list(df['acc'])
loss=list(df['loss'])

# Plot    
plt.xlabel('Epoch')
plt.ylabel('Acc   /   Loss')
plt.title('Between Epoch and Loss  /  Acc')
plt.plot(epoch,acc,'b',epoch,loss,'r')
plt.show() 

