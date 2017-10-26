# Create your first MLP in Keras
from keras.models import Sequential
from keras.layers import Dense,Dropout
from keras.callbacks import EarlyStopping  
early_stopping=EarlyStopping(monitor='loss', min_delta=0, patience=500, verbose=1, mode='auto')  

import numpy
# fix random seed for reproducibility
numpy.random.seed(1)

from pandas import read_csv

data = read_csv('train.csv')
data = data.drop(['Name','Ticket'],axis=1)

#заповнемо поле Age середнім хначенням 
data.loc[data.Age[data.Age.isnull()].index,'Age'] = data.Age[data.Age.notnull()].median()

# присвоим эти пассажирам порт в котором село больше всего людей:
MaxPassEmbarked = data.groupby('Embarked').count()['PassengerId']
data.loc[data.Embarked[data.Embarked.isnull()].index,'Embarked']=MaxPassEmbarked[MaxPassEmbarked == MaxPassEmbarked.min()].index[0]

data = data.drop(['PassengerId','Cabin'],axis=1)

from sklearn.preprocessing import LabelEncoder
label = LabelEncoder()
dicts = {}

label.fit(data.Sex.drop_duplicates()) #задаем список значений для кодирования
dicts['Sex'] = list(label.classes_)
data.Sex = label.transform(data.Sex) #заменяем значения из списка кодами закодированных элементов 

label.fit(data.Embarked.drop_duplicates())
dicts['Embarked'] = list(label.classes_)
data.Embarked = label.transform(data.Embarked)

test = read_csv('test.csv')
test.loc[test.Age[test.Age.isnull()].index,'Age'] = test.Age[test.Age.notnull()].median()
test.loc[test.Fare[test.Fare.isnull()].index,'Fare'] = test.Fare.median() #заполняем пустые значения средней ценой билета
MaxPassEmbarked = test.groupby('Embarked').count()['PassengerId']
result = test.PassengerId
test = test.drop(['Name','Ticket','Cabin','PassengerId'],axis=1)

label.fit(dicts['Sex'])
test.Sex = label.transform(test.Sex)

label.fit(dicts['Embarked'])
test.Embarked = label.transform(test.Embarked)

y= data[['Survived']]
y=y.values
X =data[['Pclass','Sex','Age','SibSp','Parch','Fare','Embarked']]
X=X.values 
    
for l1 in range(81,91):
  for l2 in range(81,91):
    
    print ("-------------------------------------- Layer 1 = ",l1,"  Layer 2 = ",l2)
    
    # create model
    model = Sequential()
    model.add(Dense(l1, input_dim=X.shape[1], activation='relu')) # 274
    model.add(Dense(l2, activation='relu')) # relu 548
    model.add(Dense(1, activation='sigmoid'))  # softmax sigmoid
    
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    # Fit the model
    model.fit(X, y, epochs=50000, batch_size=99,callbacks=[early_stopping],verbose=0)
    # validation_split=0.2, 
    
    """
    predictions = model.predict(X)
    yy=0
    for x in predictions:
        print("Test data N ",yy,' = ',round(x[0],2)," S = ",y[yy])
        yy=yy+1
    """
    
    predictions = model.predict(X)
    yy=0
    for x in predictions:
        if yy==2 or yy==17 or yy==34 or yy==35 or yy==60: 
            print(" Id = ",yy,' = ',round(x[0],2)," S = ",y[yy])
        yy=yy+1
        
    

"""
X_pre =test[['Pclass','Sex','Age','SibSp','Parch','Fare','Embarked']]
P =X_pre.values 
# print (P)
# calculate predictions
predictions = model.predict(P)
# round predictions
rounded = [round(x[0],5) for x in predictions]
print("Train data = ",rounded)
"""

# evaluate the model
#scores = model.evaluate(X, Y)
#print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

