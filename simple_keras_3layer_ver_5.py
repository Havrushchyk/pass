# Create your first MLP in Keras
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping  
early_stopping=EarlyStopping(monitor='loss', min_delta=0, patience=100, verbose=1, mode='auto')  

import numpy as np
# fix random seed for reproducibility
np.random.seed(1)

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
from sklearn.ensemble import RandomForestClassifier
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

y_test=[]

X_test =test[['Pclass','Sex','Age','SibSp','Parch','Fare','Embarked']]
X_test=X_test.values
    
# Keras 
for l1 in range(33,40): # 17 23 25 30
  #for l2 in range(81,91):
    print ("-------------------------------------- Layer 1 = ",l1,"  Layer 2 = ",l1)
    # create model    - model.summary()
    model = Sequential()
    model.add(Dense(l1, input_dim=X.shape[1], activation='relu')) # tanh
    model.add(Dense(l1, activation='relu')) # relu
    model.add(Dense(l1, activation='relu')) # relu
    model.add(Dense(l1, activation='relu')) # relu
    model.add(Dense(l1, activation='relu')) # relu
    model.add(Dense(l1, activation='relu')) # relu
    model.add(Dense(1, activation='sigmoid'))  # sigmoid
    
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy']) # sgd adam RMSprop
    # Fit the model 
    model.fit(X, y.ravel(), epochs=2500, batch_size=64,callbacks=[early_stopping],verbose=1) # validation_split=0.2, 
    
    # Evaluate
    loss, accuracy = model.evaluate(X, y, verbose=0)
    print('Keras - Loss: ', loss,' Accuracy: ', accuracy,"\n")
    
    
    # predict for TRAIN data
    keras_train = model.predict(X)
    
    keras_error=0
    keras_train_norm=keras_train
    for res in range(len(keras_train)):
        if keras_train[res]<0.5:
            keras_train_norm[res]=0
        else:
            keras_train_norm[res]=1
        if round(keras_train_norm[res][0]) != y[res]: keras_error+=1
            
    train_acc_keras=(X.shape[0]-keras_error)/X.shape[0]
    print ("Train_acc_keras = {0}".format(train_acc_keras*100)," errors = ",keras_error)
     
    # predict for TEST data
    keras_test = model.predict(X_test)        
    
    keras_test_norm=keras_test
    for res in range(len(keras_test)):
        if keras_test[res]<0.5:
            keras_test_norm[res]=0
        else:
            keras_test_norm[res]=1

    # RandomForestClassifier    
    random_forest = RandomForestClassifier(n_estimators=50,random_state=0)
    random_forest.fit(X, y.ravel())
    random_forest_test = random_forest.predict(X_test)
    random_forest_train = random_forest.predict(X)
    random_forest.score(X, y)
        
    # TRAIN
    random_forest_error=0
    random_forest_norm=random_forest_train
    for res in range(len(random_forest_train)):
        if random_forest_train[res]<0.5:
            random_forest_norm[res]=0
        else:
            random_forest_norm[res]=1
        if round(random_forest_norm[res]) != y[res]: random_forest_error+=1
            
    train_acc_random_forest=(X.shape[0]-random_forest_error)/X.shape[0]
    print ("Train_acc_random_forest = {0}".format(train_acc_random_forest*100)," errors = ",random_forest_error)

    # TEST 
    random_forest_test_norm=random_forest_test
    for res in range(len(random_forest_test)):
        if random_forest_test[res]<0.5:
            random_forest_test_norm[res]=0
        else:
            random_forest_test_norm[res]=1
        
    # TEST 1 
    random_forest_test_norm1=random_forest_test
    for res in range(len(random_forest_test)):
        if random_forest_test[res]<0.5:
            random_forest_test_norm1[res]=0
        else:
            random_forest_test_norm1[res]=1
    
    rf_test=np.concatenate((random_forest_test_norm[:, None],random_forest_test_norm1[:, None]), axis=1)

    
    # Xgboost Classifier
    import xgboost as xgb
    # read in data
    dtrain = xgb.DMatrix(X,y.ravel())
    dtest = xgb.DMatrix(X_test)
    # specify parameters via map
    param = {'max_depth':10, 'eta':1, 'silent':1, 'objective':'binary:logistic' }
    num_round = 50
    bst = xgb.train(param, dtrain, num_round)
    # make prediction
    xgboost_train = bst.predict(dtrain)
    xgboost_test = bst.predict(dtest)
    
    
    xgboost_error=0
    xgboost_train_norm=xgboost_train
    for res in range(len(xgboost_train)):
        if xgboost_train[res]<0.5:
            xgboost_train_norm[res]=0
        else:
            xgboost_train_norm[res]=1
        if round(xgboost_train_norm[res]) != y[res]: xgboost_error+=1
    
    train_acc_xgboost=(X.shape[0]-xgboost_error)/X.shape[0]
    print ("Train_acc_xgboost = {0}".format(train_acc_xgboost*100), " errors = ", xgboost_error)

    
    res_train=np.concatenate((y,keras_train,xgboost_train[:, None],random_forest_train[:, None]), axis=1)
    
    res_test=np.concatenate((keras_test,xgboost_test[:, None],random_forest_test[:, None]), axis=1)

    res_train_norm=np.concatenate((y,keras_train_norm,xgboost_train[:, None],random_forest_train[:, None]), axis=1)

data_result = read_csv('d:\DataS\coursera\gender_submission.csv')
data_result.Survived=random_forest_test_norm #.astype(int)
data_result.to_csv('d:\DataS\coursera\gender_submission.csv', index=False)
