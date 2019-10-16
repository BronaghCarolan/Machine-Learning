from sklearn.model_selection import GridSearchCV
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn import model_selection
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from category_encoders import *
import xgboost as xgb
from catboost import CatBoostRegressor
from sklearn.ensemble import RandomForestRegressor
from lightgbm import LGBMRegressor
from math import sqrt

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

#Pipeline dealing with categorical variables is set up

#Imputer step to fill in all missing values as a constant 'MISSING'
cat_si_step = ('si', SimpleImputer(strategy='constant',
                                   fill_value='MISSING'))
#Encoding step to transform categorical data to numerical data using Target Encoder (keeps data in one column)
cat_te_step = ('te', TargetEncoder())
cat_steps = [cat_si_step, cat_te_step]
cat_pipe = Pipeline(cat_steps)
#Categorical columns
cat_cols = ['Gender', 'Country', 'University Degree', 'Profession']


#Pipeline dealing with numerical variables is set up

#Numerical columns
num_cols = ['Year of Record', 'Age', 'Body Height [cm]']
#Imputer step to fill in all missing values as the median of the corresponding column
num_si_step = ('si', SimpleImputer(strategy='median'))
#Scaling all numerical data
num_ss_step = ('ss', StandardScaler())
num_steps = [num_si_step, num_ss_step]
num_pipe = Pipeline(num_steps)

#Combining the categorical and numerical pipelines using ColumnTransformer
transformers = [('cat', cat_pipe, cat_cols),
                ('num', num_pipe, num_cols)]
ct = ColumnTransformer(transformers=transformers)


#Setting regressor - XGBoost, CatBoost and LightGBM give me similar outputs

#regressor = xgb.XGBRegressor()
#regressor= CatBoostRegressor(od_type = 'Iter', od_wait = 100, verbose =0, cat_features = [2,4])
regressor = LGBMRegressor()


#Splitting data into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    train, y, test_size=0.3)
#Pipeline is set - given the ColumnTransformer (categorical and numerical) and the regressor
ml_pipe = Pipeline([ 
    ('transform', ct),
    ('regressor', regressor)])
#Fitting pipeline
ml_pipe.fit(X_train, y_train)

#Parameters are set for current regressor

'''
parameters_XGB = {
    'regressor__nthread':[4], #when use hyperthread, xgboost may become slower
    'regressor__objective':['reg:linear'],
    'regressor__learning_rate': [.03, 0.05, .07], #so called `eta` value
    'regressor__max_depth': [3,1,2,6,4,5,7,8,9,10],
    'regressor__min_child_weight': [2,3],
    'regressor__silent': [True],
    'regressor__subsample': [0.7],
    'regressor__colsample_bytree': [0.7],
    'regressor__n_estimators': [250,100,500,1000],
    'regressor__min_child_weight':[4,5], 
    'regressor__gamma':[0.3],  
    'regressor__subsample':[i/10.0 for i in range(6,11)]
}
'''
parameters_CAT = {
        #'regressor__depth':[ 4, 6,7, 8],
        #'regressor__iterations':[500, 1000],
        #'regressor__learning_rate':[0.01,0.1,0.2,0.3], 
        #'regressor__l2_leaf_reg':[3, 1, 2],
        #'regressor__border_count':[100,200],
        #'regressor__ctr_border_count':[50,5,10,20,100,200],
        #'regressor__thread_count':[4]
}
parameters_LGB = {
    'regressor__boosting_type': ['gbdt'],
    'regressor__max_bin':[50,60,70,80],
    'regressor__objective': ['regression'],
    #'regressor__metric' : {'l2','auc'},
    'regressor__num_leaves': [30],
    #'regressor__min_data' : 50,
    'regressor__max_depth' : [6],
    #'regressor__learning_rate': 0.01,
    'regressor__feature_fraction': [0.7],
   'regressor__bagging_fraction': [0.7],
    #'regressor__subsample':range(0.4, 1),
    #'regressor__bagging_freq': 80,
    'regressor__verbose': [100]
}
        


#GridSearchCV is given the pipeline, the parameters for the regressor
print('Starting grid search')
gridsearch = GridSearchCV(ml_pipe, parameters_LGB,verbose=1, cv=5).fit(X_train, y_train)
#Predict using test data
pred = gridsearch.predict(X_test)
print(np.sqrt(mean_squared_error(y_test, pred)))
print('Final score is: ', gridsearch.score(X_test, y_test))
print(gridsearch.best_params_)

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


#Predict using submission data
pred2 = gridsearch.predict(predict_X)
print(pred2)
#Write to file
test = {'Income': pred2}
df_out = pd.DataFrame(test, columns=['Income'])
df_out.to_csv("TestInc.csv")



