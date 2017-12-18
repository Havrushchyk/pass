from numpy.random import seed
seed(1)
from tensorflow import set_random_seed
set_random_seed(1)

from pandas import read_csv
import pandas as pd
import numpy as np

data_train = read_csv('train.csv')
data_test = read_csv('test.csv')

# data_test.Age=data_test.Age.fillna(data_test['Age'].median())
combined = pd.concat([data_train,data_test])

#заповнемо поле Age середнім хначенням 
combined.loc[combined.Age[combined.Age.isnull()].index,'Age'] = combined.Age[combined.Age.notnull()].median()



"""
survived_sex = data[data['Survived']==1]['Sex'].value_counts()
dead_sex = data[data['Survived']==0]['Sex'].value_counts()
df = pd.DataFrame([survived_sex,dead_sex])
df.index = ['Survived','Dead']
df.plot(kind='bar',stacked=True, figsize=(7,4))
"""
    
combined['Title'] = combined.Name.copy().astype(str)
combined['Title'] = combined['Title'].str.extract(' ([A-Za-z]+)\.', expand=False)

import re
def clean_name(name):
    # Первое слово до запятой - фамилия
    s = re.search('^[^,]+, (.*)', name)
    if s:
        name = s.group(1)
        
    # Удаляем обращения
    name = re.sub('(Mr\. |Mrs\. |Miss\. |Master\. |Don\. |Dona\. |Rev\. |Dr\. |Mme\. |Ms\. |Major\. |Lady\. |Sir\. |Col\. |Capt\. |Countess\. |Jonkheer\.)', '', name)    
    # Если есть скобки - то имя пассажира в них
    s = re.search('\(([^)]+)\)', name)
    if s:
        name = s.group(1)
        
    # Берем первое оставшееся слово и удаляем кавычки
    name = name.split(' ')[0].replace('"', '')
    
    return name

def clean_subname(subname):
    # Первое слово до запятой - фамилия
    s = re.search('^[^,]+', subname)
    subname=s.group(0)
    return subname


combined['Name1'] = combined['Name'].map(clean_name)
# name_counts = data['Name1'].value_counts()

combined['Name2'] = combined['Name'].map(clean_subname)
# subname_counts = data['Name2'].value_counts()


# a function that extracts alfabet char of the ticket, returns (i.e the ticket is a digit)
def cleanTicket(ticket):
        reg = re.compile('[^a-zA-Z ]')
        p=reg.sub('', ticket)
        if len(p)>0:
            return 1 # reg.sub('', ticket)
        else:
            return 0 #'None'
        
combined['Ticket'] = combined['Ticket'].map(cleanTicket)

#Work on titles
def title_group(data):
    data['Title'] = data['Title'].replace(['Countess', 'Dona'],'Lady')
    data['Title'] = data['Title'].replace('Mlle', 'Miss')
    data['Title'] = data['Title'].replace(['Mme', 'Ms'], 'Mrs')
    data['Title'] = data['Title'].replace(['Capt', 'Col','Don', 'Major', 'Rev', 'Jonkheer', 'Sir'], 'Officer')
    data['Title_Group'] = data['Title'].map({"Lady": 1, "Master": 2, "Miss": 3, "Mr": 4, "Mrs": 5, "Officer": 6, "Dr":7})
    data['Title_Group'] = data['Title_Group'].astype(int)
title_group(combined)

"""
def fill_age(data):
    ##### Replace missing ages, by random values compute on same title
    # for example all "Master" are young men under 13 years
    age_nan = data.loc[data["Age"].isnull()].sort_values(by='Title', ascending=True)
    age_notnan = data.loc[data["Age"].notnull()].sort_values(by='Title', ascending=True)
    title_list = ['Master','Dr','Miss','Mr','Mrs','Lady','Officer']
    for title in title_list:
        title_age = age_notnan.loc[age_notnan["Title"]==title]
        count_nan_age_title = len(age_nan[age_nan["Title"]==title])
        age_std=title_age["Age"].std()
        age_mean=title_age["Age"].mean()
        #if no std -> jus one value
        if math.isnan(age_std):
            random_age = age_mean
        else:
            random_age = np.random.randint(age_mean - age_std, age_mean + age_std, size = count_nan_age_title)

        #Replace random age values for master without age
        data.loc[((data['Age'].isnull()) & (data['Title']==title)),['Age']] = random_age
import math
fill_age(combined)
"""

# присвоим эти пассажирам порт в котором село больше всего людей:
MaxPassEmbarked = combined.groupby('Embarked').count()['PassengerId']
combined.loc[combined.Embarked[combined.Embarked.isnull()].index,'Embarked']=MaxPassEmbarked[MaxPassEmbarked == MaxPassEmbarked.min()].index[0]

# replacing missing cabins with U (for Uknown)
combined.Cabin.fillna('U', inplace=True)
# mapping each Cabin value with the cabin letter
combined['Cabin'] = combined['Cabin'].map(lambda c : c[:1]) 

combined.loc[combined.Fare[combined.Fare.isnull()].index,'Fare'] = combined.Fare.median() #заполняем пустые значения средней ценой билета

from sklearn.preprocessing import LabelEncoder
label = LabelEncoder()
dicts = {}

"""
label.fit(combined.Ticket.drop_duplicates()) #задаем список значений для кодирования
dicts['Ticket'] = list(label.classes_)
combined.Ticket = label.transform(combined.Ticket) #заменяем значения из списка кодами закодированных элементов 
"""

label.fit(combined.Name1.drop_duplicates()) #задаем список значений для кодирования
dicts['Name1'] = list(label.classes_)
combined.Name1 = label.transform(combined.Name1) #заменяем значения из списка кодами закодированных элементов 

label.fit(combined.Name2.drop_duplicates()) #задаем список значений для кодирования
dicts['Name2'] = list(label.classes_)
combined.Name2 = label.transform(combined.Name2) #заменяем значения из списка кодами закодированных элементов 

label.fit(combined.Title.drop_duplicates()) #задаем список значений для кодирования
dicts['Title'] = list(label.classes_)
combined.Title = label.transform(combined.Title) #заменяем значения из списка кодами закодированных элементов 


label.fit(combined.Cabin.drop_duplicates()) #задаем список значений для кодирования
dicts['Cabin'] = list(label.classes_)
combined.Cabin = label.transform(combined.Cabin) #заменяем значения из списка кодами закодированных элементов 

label.fit(combined.Sex.drop_duplicates()) 
dicts['Sex'] = list(label.classes_)
combined.Sex = label.transform(combined.Sex) 

label.fit(combined.Embarked.drop_duplicates())
dicts['Embarked'] = list(label.classes_)
combined.Embarked = label.transform(combined.Embarked)

combined = combined.drop(['Name'],axis=1) # 'PassengerId','Ticket',

#combined=combined['Age'].dropna()

#c_list_index=list(data_train.Age.loc[data_train.Age.isnull()].index)
#combined=combined.drop(c_list_index)

train = combined.loc[combined.Survived.notnull()].copy()
test = combined.loc[combined.Survived.isnull()].copy()

train=train.dropna()

list_features = ['Fare','Name2','Name1','Age','Title_Group','Cabin','Pclass','SibSp','Ticket','Parch','Embarked'] # ,'Title_Group','Name1','Name2'


y= train[['Survived']]
y=y.values
X = train[list_features]
X=X.values

X_test =test[list_features]
X_test=X_test.values


# Create your first MLP in Keras
from keras.models import Sequential
from keras.layers import Dense #,Dropout
from keras.callbacks import EarlyStopping, CSVLogger, ModelCheckpoint
from keras.wrappers.scikit_learn import KerasClassifier

from keras.optimizers import Adam

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold

lr=0.0001
# Keras 
for loop1 in range(198,199): # 98 128
  for loop2 in range(1,11): # Best
    print ("\n","-------------------------------------- Input layer = ",loop1," Loop = ",loop2)
    
    batch_size=99
    
    lr-=0.00001
    Koef=0.95
    #"""    
    # baseline model
    def create_baseline():
        # create model
        model = Sequential()
        model.add(Dense(loop1, input_dim=X.shape[1],activation='sigmoid')) #  kernel_initializer='random_uniform',bias_initializer='random_uniform', 
        model.add(Dense(loop1, activation='sigmoid'))
        model.add(Dense(loop1, activation='sigmoid'))
        model.add(Dense(loop1, activation='sigmoid'))
        model.add(Dense(loop1, activation='sigmoid'))
        model.add(Dense(1, activation='sigmoid'))
        # Compile model
        model.compile(metrics=['accuracy'], loss='binary_crossentropy', optimizer=Adam(lr=lr) )
        # model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model
    
    model=create_baseline()

    """
    # Cross - evaluate model
    estimator = KerasClassifier(build_fn=create_baseline, epochs=10, batch_size=99,verbose=0)
    kfold = StratifiedKFold(n_splits=5 , shuffle=True, random_state=1)
    results = cross_val_score(estimator, X, y.ravel(), cv=kfold)
    """
    
    # Fit the model
    early_stopping=EarlyStopping(monitor='loss',min_delta=0.001, patience=100, verbose=1 ,mode='auto') # min_delta=0.001,
    
    # Save logs
    csv_logger = CSVLogger('log_keras.csv', append=False, separator=',')
    
    # Save weights
    filepath="best_weights.hdf5" # ./best_weights.hdf5  
    checkpoint=ModelCheckpoint(filepath, monitor='loss', save_best_only=True, verbose=0, mode='min') # save_weights_only=False,
    callbacks_list = [early_stopping,csv_logger,checkpoint] # early_stopping,
    
    model.fit(X, y.ravel(), epochs=10000, batch_size=batch_size, verbose=0, callbacks=callbacks_list, shuffle=False) # validation_split=0.2,   callbacks=[early_stopping],
    
    #print("\n","Keras - Cross_val_score  : %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))
    
    # predict for TRAIN data
    keras_train = model.predict(X,batch_size=batch_size, verbose=0)
    
    keras_error=0
    keras_train_norm=keras_train
    for res in range(len(keras_train)):
        if keras_train[res]<Koef:
            keras_train_norm[res]=0
        else:
            keras_train_norm[res]=1
        if round(keras_train_norm[res][0]) != y[res]: keras_error+=1
            
    train_acc_keras=(X.shape[0]-keras_error)/X.shape[0]
    print ("Keras - train acc = {0:0.2f}".format(train_acc_keras*100)," errors = ",keras_error,"\n")
     
    # predict for TEST data
    keras_test = model.predict(X_test,batch_size=batch_size, verbose=0)        
    
    keras_test_norm=keras_test
    for res in range(len(keras_test)):
        if keras_test[res]<Koef:
            keras_test_norm[res]=0
        else:
            keras_test_norm[res]=1
    
    # Load best weigth
    model2 = Sequential()
    model2.add(Dense(loop1, input_dim=X.shape[1],activation='sigmoid')) # kernel_initializer='random_uniform', bias_initializer='random_uniform',
    model2.add(Dense(loop1, activation='sigmoid'))
    model2.add(Dense(loop1, activation='sigmoid'))
    model2.add(Dense(loop1, activation='sigmoid'))
    model2.add(Dense(loop1, activation='sigmoid'))
    model2.add(Dense(1, activation='sigmoid'))
    # load weights
    model2.load_weights("best_weights.hdf5")
    # Compile model
    model2.compile(metrics=['accuracy'], loss='binary_crossentropy', optimizer=Adam(lr=lr) )

    print("--- Created model and loaded weights from file ---")
    # estimate accuracy on whole dataset using loaded weights
    scores = model2.evaluate(X, y, verbose=0)
    print("%s: %.2f%%" % (model2.metrics_names[1], scores[1]*100))
    
    keras_train = model2.predict(X,batch_size=batch_size, verbose=0)
    

    
    keras_error=0
    keras_train_norm=keras_train
    for res in range(len(keras_train)):
        if keras_train[res]<Koef:
            keras_train_norm[res]=0
        else:
            keras_train_norm[res]=1
        if round(keras_train_norm[res][0]) != y[res]: keras_error+=1
            
    train_acc_keras=(X.shape[0]-keras_error)/X.shape[0]
    print ("Keras - train acc = {0:0.2f}".format(train_acc_keras*100)," errors = ",keras_error,"\n")
     
    # predict for TEST data
    keras_test = model2.predict(X_test,batch_size=batch_size, verbose=0)        
    
    keras_test_norm=keras_test
    for res in range(len(keras_test)):
        if keras_test[res]<Koef:
            keras_test_norm[res]=0
        else:
            keras_test_norm[res]=1
    #"""

    """
    
    # RandomForestClassifier    
    random_forest = RandomForestClassifier(n_estimators=250,random_state=1,min_samples_split=2,max_depth=20)
    kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=0)
    results = cross_val_score(random_forest, X, y.ravel(), cv=kfold)
    print("Random_forest  - Cross_val_score %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))
    
    random_forest.fit(X, y.ravel())
    random_forest_train = random_forest.predict_proba(X)
    random_forest_test = random_forest.predict_proba(X_test)
    importances = random_forest.feature_importances_

    
    # TRAIN
    random_forest_error=0
    random_forest_norm=random_forest_train[:,1]
    for res in range(len(random_forest_train)):
        if random_forest_train[res][1]<Koef:
            random_forest_norm[res]=0
            random_forest_norm[res]=1
        if round(random_forest_norm[res]) != y[res]: random_forest_error+=1
            
    train_acc_random_forest=(X.shape[0]-random_forest_error)/X.shape[0]
    print ("Random_forest train acc = {0:0.2f}".format(train_acc_random_forest*100)," errors = ",random_forest_error,"\n")

    # TEST 
    random_forest_test_norm=random_forest_test[:,1]
    for res in range(len(random_forest_test)):
        if random_forest_test[res][1]<Koef:
            random_forest_test_norm[res]=0
        else:
            random_forest_test_norm[res]=1

    
    # Xgboost Classifier
    import xgboost as xgb
    # read in data
    dtrain = xgb.DMatrix(X,y)
    dtest = xgb.DMatrix(X_test)
    # specify parameters via map
    params = {'max_depth':8, 'eta':1, 'silent':1, 'objective':'binary:logistic' }
    num_round = 100
    bst = xgb.train(params, dtrain, num_round)
    # make prediction
    xgboost_train = bst.predict(dtrain)
    xgboost_test = bst.predict(dtest)
    
    X_cv=xgb.cv(params, dtrain, num_boost_round=100, nfold=10,metrics='auc')
    print ("Xgboost -  Cross validation Auc = ", X_cv['test-auc-mean'].mean()*100)
    

    xgboost_error=0
    xgboost_train_norm=xgboost_train
    for res in range(len(xgboost_train)):
        if xgboost_train[res]<Koef:
            xgboost_train_norm[res]=0
        else:
            xgboost_train_norm[res]=1
        if round(xgboost_train_norm[res]) != y[res]: xgboost_error+=1
    
    train_acc_xgboost=(X.shape[0]-xgboost_error)/X.shape[0]
    print ("Xgboost train acc = {0}".format(train_acc_xgboost*100), " errors = ", xgboost_error)

    #Xboost TEST
    xgboost_test_norm=xgboost_test
    for res in range(len(xgboost_test)):
        if xgboost_test[res]<Koef:
            xgboost_test_norm[res]=0
        else:
            xgboost_test_norm[res]=1


    random_forest_trai=random_forest_train[:,1]
    random_forest_tes=random_forest_test[:,1]
    
    res_train=np.concatenate((y,keras_train_norm, xgboost_train_norm[:, None],random_forest_norm[:, None]), axis=1)
    res_test=np.concatenate((keras_test_norm, xgboost_test_norm[:, None],random_forest_test_norm[:, None]), axis=1)
"""  
        
    import matplotlib.pyplot as plt
    
    df=read_csv("log_keras.csv",usecols=['epoch','acc','loss'])
    
    epoch=list(df['epoch'])
    acc=list(df['acc'])
    loss=list(df['loss'])
    
    # Plot    
    plt.xlabel('Epoch')
    plt.ylabel('Acc   /   Loss')
    plt.title('Between Epoch and Loss  /  Acc')
    plt.plot(epoch,acc,'b',epoch,loss,'r')
    plt.show() 

    
"""
data_result = read_csv('d:\DataS\coursera\gender_submission.csv')
data_result.Survived=xgboost_test_norm.astype(int)
data_result.to_csv('d:\DataS\coursera\gender_submission_x.csv', index=False)
data_result.Survived=random_forest_test_norm.astype(int)
data_result.to_csv('d:\DataS\coursera\gender_submission_r.csv', index=False)
data_result.Survived=keras_test_norm.astype(int)
data_result.to_csv('d:\DataS\coursera\gender_submission_k.csv', index=False)


    
    import matplotlib.pyplot as plt
    
    df=read_csv("log_keras.csv",usecols=['epoch','acc','loss'])
    
    epoch=list(df['epoch'])
    acc=list(df['acc'])
    loss=list(df['loss'])
    
    # Plot    
    plt.xlabel('Epoch')
    plt.ylabel('Acc   /   Loss')
    plt.title('Between Epoch and Loss  /  Acc')
    plt.plot(epoch,acc,'b',epoch,loss,'r')
    plt.show() 
"""