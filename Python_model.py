#Problem Description
#A bike rental or bike hire business is a bicycle shop or other business that rents bikes for short periods of time (usually for a few hours) for a fee. Bike rental shops primarily serve people who don't have access to a vehicle, typically travelers and particularly tourists. Specialized bike rental shops thus typically operate at beaches, parks, or other locations those tourists frequent. In this case, the fees are set to encourage renting the bikes for a few hours at a time, rarely more than a day.

#  regrassion model for bike renting based on enviornmental condition
# importing libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# reading csv file
bike_data = pd.read_csv(r"C:\Users\anupr\Desktop\projects\bike renting\day.csv", sep = ',')


############################################
#                                          #
#     2.1 Exploratory Data Analysis        #
#                                          #
############################################

###################################
#  2.1.1 understanding the data   #
###################################
#Checking few observation of dataset
bike_data.head()

# looking at information of dataset
bike_data.info()

numeric_columns = ['temp', 'atemp', 'hum', 'windspeed', 'casual', 'registered', 'cnt']
cat_columns = ['season', 'yr', 'mnth', 'holiday', 'weekday', 'workingday', 'weathersit']

# looking at five point summary for our numerical variables
bike_data[numeric_columns].describe()

# unique values of categories variables
bike_data[cat_columns].nunique()

# counting of each unique values in each categorical variable
print("value counts of categorical column")
print()
for i in cat_columns:
    print(i)
    print(bike_data[i].value_counts())
    print("=================================")

###################################
#  2.1.2 Missing value analysis   #
###################################
    
# checking for missing values in dataset
bike_data.isnull().sum()

###################################
#  2.1.3 outlier analysis         #
###################################

# user defined function that will plot boxplot and distribution for four column of dataset
def hist_and_box_plot(col1, col2, col3, col4, data, bin1=30, bin2=30, bin3=30, bin4 = 30, sup =""):
    fig, ax = plt.subplots(nrows = 2, ncols = 4, figsize= (14,6))
    super_title = fig.suptitle("Boxplot and Histogram: "+sup, fontsize='x-large')
    plt.tight_layout()
    sns.boxplot(y = col1, data = data, ax = ax[0][0])
    sns.boxplot(y = col2,data = data, ax = ax[0][1])
    sns.boxplot(y = col3, data = data, ax = ax[0][2])
    sns.boxplot(y = col4, data = data, ax = ax[0][3])
    sns.distplot(data[col1], ax = ax[1][0], bins = bin1)
    sns.distplot(data[col2], ax = ax[1][1], bins = bin2)
    sns.distplot(data[col3], ax = ax[1][2], bins = bin3)
    sns.distplot(data[col4], ax = ax[1][3], bins = bin4)
    fig.subplots_adjust(top = 0.90)
    plt.show()

# plotting boxplot and histogram for our numerical variables
hist_and_box_plot('temp', 'atemp', 'hum', 'windspeed', bin1 = 10, data = bike_data)

###################################
#  2.1.4 Feature Engineering      #
###################################

# user defined function to plot bar plot of a column for each y i.e. y1 and y2 wrt 
# unique variables of each x i.e. x1 and x2
# X1 and X2 would be categorical variable, y1 and y2 would be continuous
# this funciton will plot barplot for y1 column for each unique values of x1 and 
# will do barplot for y2 for each unique values of x2 and method could be mean,sum etc.
def plot_bar(x1, y1,x2, y2, method = 'sum'):
    fig, ax = plt.subplots(nrows = 1, ncols = 2, figsize= (12,4), squeeze=False)
    super_title = fig.suptitle("Bar Plot ",  fontsize='x-large')
    if(method == 'mean'):
        gp = bike_data.groupby(by = x1).mean()
        gp2 = bike_data.groupby(by = x2).mean()
    else:
        gp = bike_data.groupby(by = x1).sum()
        gp2 = bike_data.groupby(by = x2).sum()
    gp = gp.reset_index()
    gp2 = gp2.reset_index()
    sns.barplot(x= x1, y = y1, data = gp, ax=ax[0][0])
    sns.barplot(x= x2, y = y2, data = gp2, ax=ax[0][1])
    fig.subplots_adjust(top = 0.90)
    plt.show()
	

# plotting barplot for count i.e. cnt wrt to yr and month
plot_bar('yr', 'cnt', 'mnth', 'cnt')


#plotting barplot for count wrt month for each year
gp = bike_data.groupby(by = ['yr', 'mnth']).sum().reset_index()
sns.factorplot(x= 'mnth', y = 'cnt', data = gp, col = 'yr', kind = 'bar')

# ploting barplot for counting wrt weekday
plot_bar('weekday', 'cnt', 'weekday', 'cnt')

# defining function for making bins in mnth and weekday
def feat_month(row):
    if row['mnth'] <= 4 or row['mnth'] >=11:
        return(0)
    else:
        return(1)
    
def feat_weekday(row):
    if row['weekday'] < 2:
        return(0)
    else:
        return(1)


bike_data['month_feat'] = bike_data.apply(lambda row : feat_month(row), axis=1)
bike_data = bike_data.drop(columns=['mnth'])
bike_data['week_feat'] = bike_data.apply(lambda row : feat_weekday(row), axis=1)
bike_data = bike_data.drop(columns=['weekday'])


###################################
#  2.1.5 Feature Selection        #
###################################

# let us look at correlation plot for each numerical variables
sns.pairplot(bike_data[numeric_columns])

# let us look at heat map for each numerical variable
# with correlation 
fig = plt.figure(figsize = (14,10))
corr = bike_data[numeric_columns].corr()
sn_plt = sns.heatmap(corr, mask = np.zeros_like(corr, dtype = np.bool), square = True,
            annot= True, cmap = sns.diverging_palette(220, 10, as_cmap= True))
plt.title("HeatMap of correlation between numerical columns of bike rental dataset")
fg = sn_plt.get_figure()
fg.savefig('heatmap.png')

# chi-square test for each categorical variable
cat_columns = ['season', 'yr', 'month_feat', 'holiday', 'week_feat', 'workingday', 'weathersit']
# making every combination from cat_columns
factors_paired = [(i,j) for i in cat_columns for j in cat_columns] 
# doing chi-square test for every combination
p_values = []
from scipy.stats import chi2_contingency
for factor in factors_paired:
    if factor[0] != factor[1]:
        chi2, p, dof, ex = chi2_contingency(pd.crosstab(bike_data[factor[0]], 
                                                    bike_data[factor[1]]))
        p_values.append(p.round(3))
    else:
        p_values.append('-')
p_values = np.array(p_values).reshape((7,7))
p_values = pd.DataFrame(p_values, index=cat_columns, columns=cat_columns)
print(p_values)


# checking importance of feature
drop_col = ['cnt', 'instant','dteday', 'registered', 'casual']
from sklearn.ensemble import ExtraTreesRegressor
reg = ExtraTreesRegressor(n_estimators=200)
X = bike_data.drop(columns= drop_col)
y = bike_data['cnt']
reg.fit(X, y)
imp_feat = pd.DataFrame({'Feature': bike_data.drop(columns=drop_col).columns,
                         'importance':reg.feature_importances_})
imp_feat.sort_values(by = 'importance', ascending=False).reset_index(drop = True)


# checking vif of numerical column withhout dropping multicollinear column
from statsmodels.stats.outliers_influence import variance_inflation_factor as vf             
from statsmodels.tools.tools import add_constant
numeric_df = add_constant(bike_data[['temp', 'atemp', 'hum', 'windspeed']])
vif = pd.Series([vf(numeric_df.values, i) for i in range(numeric_df.shape[1])], 
                 index = numeric_df.columns)
vif.round(1)

# Checking VIF values of numeric columns after dropping multicollinear column i.e. atemp
from statsmodels.stats.outliers_influence import variance_inflation_factor as vf             
from statsmodels.tools.tools import add_constant
numeric_df = add_constant(bike_data[['temp', 'hum', 'windspeed']])
vif = pd.Series([vf(numeric_df.values, i) for i in range(numeric_df.shape[1])], 
                 index = numeric_df.columns)
vif.round(1)


# Making dummies for each category session and weather
season_dm = pd.get_dummies(bike_data['season'], drop_first=True, prefix='season')
bike_data = pd.concat([bike_data, season_dm],axis=1)
bike_data = bike_data.drop(columns = ['season'])
weather_dm = pd.get_dummies(bike_data['weathersit'], prefix= 'weather',drop_first=True)
bike_data = pd.concat([bike_data, weather_dm],axis=1)
bike_data = bike_data.drop(columns= ['weathersit'])


###################################
#  2.1.7 Data after EDA           #
###################################

# creating another dataset with dropping outliers i.e. bike_data_wo
bike_data_wo = bike_data.copy()
# dropping outliers from boxplot method
for i in ['windspeed', 'hum']:
    q75, q25 = np.percentile(bike_data_wo.loc[:,i], [75 ,25])
    iqr = q75 - q25
    min = q25 - (iqr*1.5)
    max = q75 + (iqr*1.5)
    bike_data_wo = bike_data_wo.drop(bike_data_wo[bike_data_wo.loc[:,i] < min].index)
    bike_data_wo = bike_data_wo.drop(bike_data_wo[bike_data_wo.loc[:,i] > max].index)

# dropping unwanted columns from both dataset bike_data and bike_data_wo
bike_data.drop(columns=['instant', 'dteday', 'holiday', 'atemp', 'casual', 'registered'], inplace=True)
bike_data_wo.drop(columns=['instant', 'dteday', 'holiday', 'atemp', 'casual', 'registered'], inplace=True)

bike_data.head()
bike_data_wo.head()

print('shape of original dataset',bike_data.shape)
print('shape of dataset after removing outliers', bike_data_wo.shape)


############################################
#                                          #
#                                          #
#   2.2.2 Building models                  #
#                                          #
#                                          #
############################################

# making a function which will build model on training set and would show result
# for k-fold cv explained_variance_score and also show result for training and test dataset
from sklearn.metrics import explained_variance_score
from sklearn.model_selection import cross_val_score
def fit_predict_show_performance(regressor, X_train, y_train, X_test, y_test):
    regressor.fit(X_train, y_train)
    y_pred = regressor.predict(X_test)
    ten_performances = cross_val_score(estimator=regressor, X = X_train, y = y_train, cv = 10, 
                                       scoring='explained_variance')
    k_fold_performance = ten_performances.mean()
    print("K-fold (K = 10) explained variance")
    print("================================")
    print(k_fold_performance)
    print()
    print("on train data explained variance")
    print("================================")
    print(regressor.score(X_train, y_train)) 
    print()
    print("on test data explained variance")
    print("================================")
    print(regressor.score(X_test, y_test))

# splitting dataset in train and test for whole dataset after eda i.e. bike_data
X = bike_data.drop(columns=['cnt'])
y = bike_data['cnt']
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# splitting dataset in train and test for dataset without outlier after eda i.e. bike_data_wo
X = bike_data_wo.drop(columns=['cnt'])
y = bike_data_wo['cnt']
from sklearn.model_selection import train_test_split
X_train_wo, X_test_wo, y_train_wo, y_test_wo = train_test_split(X, y, test_size = 0.2, random_state = 0)


#########################
#   Linear Regression   #
#########################

# building model for dataset bike_data
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
fit_predict_show_performance(regressor, X_train, y_train, X_test, y_test)

# building model for dataset bike_data_wo i.e. without  outliers
regressor = LinearRegression()
fit_predict_show_performance(regressor, X_train_wo, y_train_wo, X_test_wo, y_test_wo)


#########################
#         KNN           #
#########################

# building model for dataset bike_data
from sklearn.neighbors import KNeighborsRegressor
regressor = KNeighborsRegressor(n_neighbors=5)
fit_predict_show_performance(regressor, X_train, y_train, X_test, y_test)

# building model for dataset bike_data_wo i.e. without outliers
regressor = KNeighborsRegressor(n_neighbors=5)
fit_predict_show_performance(regressor, X_train_wo, y_train_wo, X_test_wo, y_test_wo)


#########################
#        SVM            #
#########################

# building model for dataset bike_data
from sklearn.svm import SVR
regressor = SVR()
fit_predict_show_performance(regressor, X_train, y_train, X_test, y_test)

# building model for dataset bike_data_wo i.e. without outliers
regressor = SVR()
fit_predict_show_performance(regressor, X_train_wo, y_train_wo, X_test_wo, y_test_wo)


#############################
# Decision Tree Regression  #
#############################

# building model for dataset bike_data
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state=1)
fit_predict_show_performance(regressor, X_train, y_train, X_test, y_test)

# building model for dataset bike_data_wo i.e. without outliers
fit_predict_show_performance(regressor, X_train_wo, y_train_wo, X_test_wo, y_test_wo)


#########################
#  Random Forest        #
#########################

# building model for dataset bike_data
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(random_state=1)
fit_predict_show_performance(regressor, X_train, y_train, X_test, y_test)

# building model for dataset bike_data_wo i.e. without outliers
regressor = RandomForestRegressor(random_state=1)
fit_predict_show_performance(regressor, X_train_wo, y_train_wo, X_test_wo, y_test_wo)


#########################
#     XGBRegressor      #
#########################

# building model for dataset bike_data
from xgboost import XGBRegressor
regressor = XGBRegressor(random_state=1)
fit_predict_show_performance(regressor, X_train, y_train, X_test, y_test)

# building model for dataset bike_data_wo i.e. without outliers
regressor = XGBRegressor(random_state=1)
fit_predict_show_performance(regressor, X_train_wo, y_train_wo, X_test_wo, y_test_wo)

############################################
#                                          #
#                                          #
#        Hyperparameter tuning             #
#                                          #
#                                          #
############################################
###############################################
#                                             #
# tuning Random Forest for bike_data dataset  #
#                                             #
###############################################

from sklearn.model_selection import GridSearchCV
# Random Forest hyperparameter tuning
regressor = RandomForestRegressor(random_state=1)
params = [{'n_estimators' : [500, 600, 800],'max_features':['auto', 'sqrt', 'log2'],
           'min_samples_split':[2,4,6],'max_depth':[12, 14, 16],'min_samples_leaf':[2,3,5],
           'random_state' :[1]}]
grid_search = GridSearchCV(estimator=regressor, param_grid=params,cv = 5,
                           scoring = 'explained_variance', n_jobs=-1)
grid_search = grid_search.fit(X_train, y_train)
print(grid_search.best_params_)

# building Ranodm Forest on tuned parameter
regressor = RandomForestRegressor(random_state=1, max_depth=14, n_estimators=600,
                                  max_features='auto', min_samples_leaf=2,min_samples_split=2)
fit_predict_show_performance(regressor, X_train, y_train, X_test, y_test)

###############################################
#                                             #
# tuning XGBRegressor for bike_data dataset   #
#                                             #
###############################################
regressor = XGBRegressor(random_state=1)
params = [{'n_estimators' : [250, 300,350, 400,450], 'max_depth':[2, 3, 5], 
           'learning_rate':[0.01, 0.045, 0.05, 0.055, 0.1, 0.3],'gamma':[0, 0.001, 0.01, 0.03],
           'subsample':[1, 0.7, 0.8, 0.9],'random_state' :[1]}]
grid_search = GridSearchCV(estimator=regressor, param_grid=params,cv = 5,
                           scoring = 'explained_variance', n_jobs=-1)
grid_search = grid_search.fit(X_train, y_train)
print(grid_search.best_params_)

# Building XGBRegressor on tuned parameter
regressor = XGBRegressor(random_state=1, learning_rate=0.05, max_depth=3, n_estimators=300, 
                         gamma = 0, subsample=0.8)
fit_predict_show_performance(regressor, X_train, y_train, X_test, y_test)

# plotting scatter graph for y_true and y_pred for tuned XGBRegressor model
regressor = XGBRegressor(random_state=1, learning_rate=0.05, max_depth=3, n_estimators=300, 
                         gamma = 0, subsample=0.8)
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)


fig, ax = plt.subplots(figsize=(7,5))
ax.scatter(y_test, y_pred)
ax.plot([0,8000],[0,8000], 'r--', label='Perfect Prediction')
ax.legend()
plt.title("scatter Graph between y_true and y_pred")
plt.xlabel("y_true")
plt.ylabel("y_pred")
plt.tight_layout()
plt.show()

