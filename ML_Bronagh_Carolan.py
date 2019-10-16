import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from category_encoders import *
from catboost import CatBoostRegressor
from math import sqrt
from bayes_opt import BayesianOptimization

#Loading in training dataset using pandas
dataset = pd.read_csv(
    'tcd ml 2019-20 income prediction training (with labels).csv')

#Split dataset into target(y) and predictor variables(train)
train = dataset
y = train.pop('Income in EUR').values
#Feature selection - no change to score when these are added/removed
train.pop('Instance')
train.pop('Size of City')
train.pop('Wears Glasses')
train.pop('Hair Color')

#Imputer step to fill in all categorical columns with missing values as a constant 'MISSING'
si = SimpleImputer(strategy='constant',
                                   fill_value='MISSING')
train[['Gender','Country', 'Profession','University Degree']] = si.fit_transform(train[['Gender','Country', 'Profession','University Degree']])
#Encoding step to transform categorical data to numerical data using Target Encoder (keeps data in one column)
te = TargetEncoder()
train[['Gender','Country', 'Profession','University Degree']] = te.fit_transform(train[['Gender','Country', 'Profession','University Degree']], y)

#Imputer step to fill in all missing numerical values as the median of the corresponding column
si_num = SimpleImputer(strategy='median')
train[['Year of Record', 'Age', 'Body Height']] = si_num.fit_transform(train[['Year of Record', 'Age', 'Body Height [cm]']])
#Scaling all numerical data
ss = StandardScaler()
train[['Year of Record', 'Age', 'Body Height']] = ss.fit_transform(train[['Year of Record', 'Age', 'Body Height [cm]']])

#Splitting data into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    train, y, test_size=0.2)

   
#Function to train model with parameters of function being the parameters of the regressor
def train_model(max_depth, 
                n_estimators,
                l2_leaf_reg, 
                learning_rate,
                border_count 
                ):
    params = {
        'max_depth': int(max_depth),
        'border_count': int(border_count),
        'verbose': 0,
        'l2_leaf_reg': int(l2_leaf_reg),
        'learning_rate':learning_rate,
        'n_estimators' : int(n_estimators),
        'od_type' :'Iter', #OverFitting Detector
        'od_wait' :100
    }
    #Set regressor with appropriate parameters
    model = CatBoostRegressor(**params)
    #Fit model
    model.fit(X_train, y_train)
    #Predict y_train and return -rmse to maximise
    pred = model.predict(X_train)
    return -np.sqrt(mean_squared_error(y_train, pred))


#Set params for regressor
bounds = {
    'max_depth':(5,10),
    'learning_rate': (0.01, 0.02),
    'l2_leaf_reg':(1,3),
    'n_estimators' : (200, 1000),
     'border_count':(100,200)
}

#Set Bayesian Optimizer with function for training model and the parameters
optimizer = BayesianOptimization(train_model, bounds)
#Get best parameters
optimizer.maximize(init_points=10, n_iter=50)
params = optimizer.max['params']

#Use best parameters for prediction model
model = CatBoostRegressor(max_depth= int(params['max_depth']), n_estimators = int(params['n_estimators']), l2_leaf_reg = int(params['l2_leaf_reg']), 
                                learning_rate = params['learning_rate'],  od_type ='Iter', od_wait =100)
model.fit(X_train, y_train)

pred = model.predict(X_test)
print(np.sqrt(mean_squared_error(y_test, pred)))

#load in test data using pandas
predict_df = pd.read_csv(
    './tcd ml 2019-20 income prediction test (without labels).csv')

#Split into target and predictor variables
predict_X = predict_df
predict_y = predict_X.pop('Income').values
#Feature selection to be aligned with training data
predict_X.pop('Size of City')
predict_X.pop('Wears Glasses')
predict_X.pop('Hair Color')
predict_X.pop('Instance')

#Categoical variables treated to be aligned with training data (Imputer and TargetEncoder)
predict_X[['Gender','Country', 'Profession','University Degree']] = si.transform(predict_X[['Gender','Country', 'Profession','University Degree']])
predict_X[['Gender','Country', 'Profession','University Degree']] = te.transform(predict_X[['Gender','Country', 'Profession','University Degree']], predict_y)

#Numerical variables treated to be aligned with training data (Imputer and StandardScalar)
predict_X[['Year of Record', 'Age', 'Body Height']] = si_num.transform(predict_X[['Year of Record', 'Age', 'Body Height [cm]']])
predict_X[['Year of Record', 'Age', 'Body Height']] = ss.transform(predict_X[['Year of Record', 'Age', 'Body Height [cm]']])

#Predict and store in csv file
pred2 = model.predict(predict_X)
test = {'Income': pred2}
df_out = pd.DataFrame(test, columns=['Income'])
df_out.to_csv("tcd ml 2019-20 income prediction submission file.csv")
