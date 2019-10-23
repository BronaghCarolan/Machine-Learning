# Machine-Learning
Machine Learning TCD Module Repository
ML_Bronagh_Carolan.py is the code that gave the best result on Kaggle (by a few hundred). Regressor used is CatBoost.
Bayesian Optimization is given various parameters to select the best. 

Pipeline.py gave me similar results and uses a pipeline. CatBoost gave the best result in this.

Information given in Excel Spreadsheet:

Local RMSE : 55,183

ML Libraries: scikit-learn, CatBoost

Best performing algorithm: CatBoostRegressor (RandomForestRegressor gave similar results)

Preprocessing libraries: NumPy and Pandas, category_encoders, bayesian_optimization

Feature Selection Method: Trial and error seeing if adding or removing features made a difference to the rmse

Removed Features: Instance, Hair Color, Wears Glasses, Size of City

Feature Scaling: Standard Scalar for numerical values

Feature Encoding: Target Encoding

Missing Values: Simple Imputer - for categorical variables I replace them with 'MISSING'. For numerical variables I replace them with the median

Outlier Detection: None

Data Split: 80-20 Split

Additional Steps: I started using a pipeline, using Columntransformer, to transform my data as mentioned previosly. Then used GridSearchCV to tune my hyperparamters for my regressor. I then moved to Bayesian Optimization doing the transforming of the data manually without the pipeline.

Most Important Steps: Using a pipeline brought my score from 100k down to 80k. I started with Ordinal Encoding which was not giving me great results so I switched to One-Hot whcih improved it. This made the dataset really large however so I switched from One-Hot-Encoding to Target Encoding to reduce the dimensionality which improved the run time and score massively. I started with linear regression, moved to RidgeCV, then Random Forest, then CatBoost which progressively improved my score. Lasso had a negative impact on the rmse. Tried feature selection using PCA which didn't improve anything. I found the automatic feature selection to be not much benefit. I gave some statstic tests for feature selection a brief look but also didn't find them extremely helpful. I tried splitting the Profession column into 'Senior' and 'non-Senior' people which negatively efffected the rmse. Tried doing KFoldCrossValidation on its own but using GridSearcCV gave the sameresult. Baysian optimization improved my rmse by 1000 locally and on Kaggle. I also used the CatBoost overfitting detector which stopped the iteration if overfitting occurred which made the score more accurate.






