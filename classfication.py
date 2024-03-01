#!/usr/bin/env python
# coding: utf-8

# # Table of Content
#   * [1. Getting Ready](#1.-Getting-Ready)
#   * [2. Sanity Checking & Data Cleaning](#2.-Sanity-Checking-&-Data-Cleaning)
#       - [2.1. Sanity Checking](#2.1.-Sanity-Checking)
#       - [2.2. Data Cleaning](#2.2.-Data-Cleaning)
#   * [3. Exploratory Data Analysis](#3.-Exploratory-Data-Analysis)
#       - [3.1. Univarate Data Analysis I](#3.1.-Univarate-Data-Analysis-I)
#       - [3.2. What do we learn from above bar plots?](#3.2.-What-do-we-learn-from-above-bar-plots?)
#       - [3.3 Univarate data analysis II](#3.3-Univarate-data-analysis-II)
#       - [3.4 What do we learn from above histograms?](#3.4-What-do-we-learn-from-above-histograms?)
#       - [3.5 Boxplots for Age, Duration and Campaign](#3.5-Boxplots-for-Age,-Duration-and-Campaign)
#       - [3.6 Bivarate data analysis ](#3.6-Bivarate-data-analysis)
#   * [4. Data Preprocessing](#4.-Data-Preprocessing)
#       - [4.1 Split data into train and test](#4.1-Split-data-into-train-and-test)
#       - [4.2 Handle data Multicollinearity](#4.2-Handle-data-Multicollinearity)
#       - [4.3 Detecting and treating on outliers](#4.3-Detecting-and-treating-on-outliers)
#       - [4.4 Feature Engineering on Categorical Data](#4.4-Feature-Engineering-on-Categorical-Data)
#       - [4.5 Use SMOTE to Balance the dataset](#4.5-Use-SMOTE-to-Balance-the-dataset)
#   * [5.  Staring from Logistic Regression](#5.-Staring-from-Logistic-Regression)
#   * [6.  Random Forest](#6.-Random-Forest)
#   * [7.  KNN](#7.-KNN)
#   * [8.  Stochastic Gradient Boosting](#8.-Stochastic-Gradient-Boosting)
#   * [9.  Model Evaluation](#9.-Model-Evaluation)

# # 1. Getting Ready

# In[2]:


# Basic 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as mpatches
get_ipython().run_line_magic('matplotlib', 'inline')
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

# Statistic models and Machine Learning
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.svm import SVC, LinearSVC
from sklearn.inspection import permutation_importance
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from imblearn import under_sampling, over_sampling
from imblearn.over_sampling import SMOTE


#Model Evaluation
from numpy import loadtxt
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score


# In[3]:


# Data Loading
df = pd.read_csv('Bank.csv')


# In[4]:


# shape and data types of the data
print(df.shape)
#print(df.dtypes)

# Looking at the first ten rows/basic descriptive statistics of the dataset
#print(df.head(10))


# In[5]:


df.describe()


# # 2. Sanity Checking & Data Cleaning

# ## 2.1. Sanity Checking
# *  What are the sanity checking rules for the banking data set?
# 
# > 1)If a customer was not previously contacted in the last campaign, then the outcome of last campaign to this customer should be "nonexistent";
# 
# > 2) If a customer was previously contacted in the last campaign, then the outcome of last campaign to this customer should be "success" or "failure";
# 
# > 3) If the outcome of this campaign is "yes" to a client, then the duration of contact should be greater than 0;
# 
# > 4) Some numeric values should be greater or equal to zero. They are: "Age", "duration", "campaign", "pdays", "previous" and "nr.employeed";
# 
# > 5) There should be no duplicated rows or missing values in the data.

# In[6]:


############Sanity Checking on Previous Campaign related columns############################
##First, for all records, when pdays = 999, previous should be equal to 0:
pday_999 = df.loc[(df['pdays'] == 999) & (df['previous'] != 0) ]
#print(pday_999) # There are 4110 erroneous records based on the above condition;
pday_999.y.value_counts()    
#There are 532 records are related to positive labels on our target, 
#so instead of dropping them, we make a more flexible assumption：column pdays is missing even though poutcome is available
#Let's create an indicator variable for missing 'pdays'. 
df["pdays_in"] = np.where((df['pdays'] == 999) & (df['previous'] != 0), 1, 0)
df.pdays_in.value_counts()
print(df.shape) #one more column was added.


# In[7]:


##Second, when pdays = 999, previous = 0, poutcome should be nonexistent: 
pday_none = df.loc[(df['pdays'] == 999) & (df['previous'] == 0) ]
#print(pday_none)
rslt_df_1 = pday_none.loc[(pday_none['poutcome'] != 'nonexistent')]
#print(rslt_df_1)   #No erroneous records found.

##Lastly, when pdays != 999 and previous != 0, poutcome should be "success" or 'failure":
pday_yes = df.loc[(df['pdays'] != 999) & (df['previous'] != 0) ]
#print(pday_yes)
options = ['success', 'failure']
rslt_df_2 = pday_yes.loc[~pday_yes['poutcome'].isin(options)]
print(rslt_df_2)  #No erroneous records found.


# In[8]:


############Sanity Checking on other columns############################
##If the outcome of this campaign is "yes" to a client, then the duration of contact should be greater than 0;
result_yes =df.loc[(df['y'] == 'yes')]
#print(result_yes)
rslt_df_3 = result_yes.loc[(result_yes['duration'] == 0)]
#print(rslt_df_3)   #No erroneous records found.

##Some numeric values should be greater or equal to zero. They are: "Age", "duration", "campaign", "pdays", "previous" and "nr.employeed";
rslt_df_4 = df[(df['age'] <= 0) | (df['duration'] < 0) | (df['campaign'] < 0) | (df['pdays'] < 0) | (df['previous'] < 0) | (df['nr.employed'] < 0)]
#print(rslt_df_4)  #No erroneous records found.


# ## 2.2. Data Cleaning

# In[9]:


#Any missing values? 
df.isnull().sum()
#no numerical missing values were found;
# There are "unknown" values in categorical variables, instead of dropping them, we will treat them as a sperate category;


# In[10]:


# Drop duplicates before move to the next step!!
df = df.drop_duplicates()
df.shape


# # 3. Exploratory Data Analysis

# ## 3.1. Univarate Data Analysis I
# *   Bar Plots for Categorical Variables:

# In[11]:


####################First, take a look on distribution of the response variable################################
sns.countplot(x = 'y', data = df, palette = 'hls')
plt.show()
plt.savefig('count_plot') 
# two classes of our response variable are very imbalanced, we need to fix this later;
df.groupby('y').mean() 
35000/41176


# In[12]:


#univariate analysis of categorical variables
#select non numeric columns
df_non_numeric = df.select_dtypes(exclude=[np.number])
non_numeric_cols = df_non_numeric.columns.values
print(non_numeric_cols)


# In[13]:


#plot the bar graph for non-numerical variables use one function
for column in non_numeric_cols:
    plt.figure(figsize=(15,8))
    large_to_small = df.groupby(column).size().sort_values().index[::-1]
    df[column].value_counts().plot(kind="bar",rot=45, color="b", 
                                           grid = 'true',
                                           figsize=(8,6))
    plt.xlabel(column,fontsize=16)
    plt.ylabel("# of Clients",fontsize=16)


# ## 3.2. What do we learn from above bar plots?
# * job：The three top job titles in clients are: admin, blue-collar and technician;
# * marital： More than half of the clients are married;
# * education： Top two categories of clients education levels are University degree and high school degree holders;
# * default: Only 3 clients have default credit history in banking data--might not be a great predictor;
# * housing: The number of clients who have housing loan are pretty similar with the number of clients who don't have a housing loan;
# * loan: More than 80% of clients do not have a personal loan;
# * contact: More than 80% of clients were contacted through cellular phone;
# * month: Customers were mostly contacted on May, followed by July, then August;
# * day_of_week: With almost evenly distributed curve, Thursday made the most calls to clients;
# * poutcome: Poutcome indicates the result of last campagin for customers; More than 85% of customers were not contacted in previous campagin, so the result is non existent;
# * y: Two classes of our response variable are very imbalanced, we need to fix this later; (More than 85% of clients said no to this product)

# ## 3.3 Univarate data analysis II
# *   Histograms for numerical/continious variables:

# In[14]:


# select numeric columns
df_numeric = df.select_dtypes(include=[np.number])
numeric_cols = df_numeric.columns.values
print(numeric_cols)

#First, make a function to plot all distributions for numerical columns:
plt.style.use("ggplot")
for column in numeric_cols:
    plt.figure(figsize=(20,4))
    plt.subplot(121)
    sns.distplot(df[column], kde=True)
    plt.xlabel(column)
    plt.ylabel("# of Clients")
    plt.title("Univariate Analysis of" + " " + column)

#Obviously, there are outliers detected for "duration","campagin",and "previous" 
#because all three of distributions are left-skewed 
#--recall "pdays_in" is an indicator created before.


# In[15]:


# change the response variable to binary intergers (no for 0, yes for 1)
df['y'].isnull().values.any()
df['y'].replace(['no', 'yes'], [0, 1], inplace=True)
#print(df.dtypes)
df['y'].value_counts()


# In[16]:


#barplot of variable "age": use pd.cut to define group of age

out = pd.cut(df.age, bins=[0,25,30,35,40,45,50,55,60,65,70,90], include_lowest=True)
ax = out.value_counts(sort=False).plot.bar(rot=30, color="b", 
                                           grid = 'true',
                                           figsize=(8,6))
ax.set_xticklabels(['(17-25]','(25-30]','(30-35]','(35-40]','(40-45]','(45-50]',
                            '(50-55]','(55-60]','(60-65]','(65-70]','>70'])
ax.set_xlabel('Age',fontsize=16)
ax.set_ylabel('# of clients',fontsize=16)
ax.xaxis.set_tick_params(labelsize=12)
ax.yaxis.set_tick_params(labelsize=12)
plt.show()


# ## 3.4 What do we learn from above histograms?
# * age: Clients mostly concentrated around the 20 to 60 value range, several outliers at the end.
# * Pdays: records how many days have been since the last contact of this client, 999 indicates the customer was never contacted before;More than 95% of customers were never contacted before;
# * Previous: indicates how many times the customer was contacted in previous campaign; More than 85% of customers were contacted zero times in previous campagin; 
# * Duration and campaign are heavily skewed and this might due to the presence of outliers.
# * emp.var.rate, cons.price.idx, cons.conf.idx, euribor3m, nr.employed: no obvious pattern found.

# ## 3.5 Boxplots for Age, Duration and Campaign
# * This is for further detect the outliers for above three features

# In[17]:


#boxplot for age, duration and campagin:
plt.style.use("ggplot")
for column in ["age", "duration", "campaign"]:
    plt.figure(figsize=(20,4))
    plt.subplot(121)
    sns.boxplot(df[column])
    plt.title(column)
#Note that we must drop 'duration' column for the final modeling, 
#thus we don't need to treat the outliers


# * There are indeed outliers on the right tail of distributions of above three features, we will deal with it later at pre-processing stage of data set.

# ## 3.6 Bivarate data analysis 

# In[18]:


#defining bivarate plot:
def plot_bivarate(col):
    plt.figure()
    x1 = df.groupby(col)[col].count()
    x2 = df.groupby(col).y.mean()
    ax1 = x1.plot(kind = 'bar',rot=45, 
                  color="b", grid = 'true',figsize=(8,6))
    ax1.set_xlabel(col,fontsize=16)
    ax1.set_ylabel('# of clients',fontsize=16)
    ax1.xaxis.set_tick_params(labelsize=10)
    ax1.yaxis.set_tick_params(labelsize=10)

    ax2 = ax1.twinx()
    x2. plot(kind = 'line', ax = ax2, color = 'orange', lw = 3)
    ax2.set_ylabel('% Conversion Rate',fontsize=16)
    ax2.set_ylim(0, 1)
    ax2.set_xticklabels(ax1.get_xticklabels(), 
                          rotation=45, 
                          horizontalalignment='right')


# In[19]:


#Bivariate analysis for categorical variables:
for col in non_numeric_cols:
    plot_bivarate(col)


# In[20]:


#Bivariate plot for variable "age":
plt.figure()
x1 = out.value_counts(sort=False)
x2 = df.groupby(out).y.mean()
ax1 = x1.plot.bar(rot=30, color="b", grid = 'true',figsize=(8,6))
ax1.set_xticklabels(['(17-25]','(25-30]','(30-35]','(35-40]','(40-45]','(45-50]',
                            '(50-55]','(55-60]','(60-65]','(65-70]','>70'])
ax1.set_xlabel('Age',fontsize=16)
ax1.set_ylabel('# of clients',fontsize=16)
ax1.xaxis.set_tick_params(labelsize=12)
ax1.yaxis.set_tick_params(labelsize=12)

ax2 = ax1.twinx()
x2. plot(kind = 'line', ax = ax2, color = 'orange', lw = 3)
ax2.set_ylabel('% Conversion Rate',fontsize=16)
ax2.set_ylim(0, 1)
ax2.set_xticklabels(ax1.get_xticklabels(), 
                          rotation=45, 
                          horizontalalignment='right')


# # 4. Data Preprocessing
# *   Split data into train and test;
# *   Handle data Multicollinearity;
# *   Deal with Outliers;
# *   One hot encoding/dummy variables--turn categorical variables into numerical variable;
# *   Variable Selection-dimension reduction;
# *   Balance the dataset (SMOTE)

# ## 4.1 Split data into train and test

# In[21]:


##################Correlation Matrix##########################
# Use the correlation matrix to decide what to do next:
corr = df.corr()
corr.style.background_gradient(cmap='PuBu')


# **From correlation matrix we observe next:**
# 
# 1.   Most correlated with target feature is call duration. So we need to remove it to build a realistic predictive model as the dataset dictionary suggested;
# 2.   Besides "duration", the top 5 correlated numerical variables with our response are: "nr.employed", "pdays", "euribor3m", "emp.var.rate" and "previous"; 
# 3. The emp.var.rate, cons.price.idx, euribor3m and nr.employed features have very high correlation. With euribor3m and nr.employed having the highest correlation of 0.95! We need to drop one of them from each highly correlated combinations to avoid multicollinearity.

# In[22]:


#Remove "duration" column:
df = df.drop(columns=['duration'], axis = 1)


# In[23]:


# Split the data into X and y
X = df.drop("y",axis=1)
y = df["y"]

## Split the data into trainx, testx, trainy, testy with test_size = 0.30
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=0)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape) 


# In[24]:


#check the distribution of y_train, y_test to verify the split results:
y_train.value_counts()
y_test.value_counts() #still imbalanced on response variable, good!


# ## 4.2 Handle data Multicollinearity
# *  Calculate VIF and remove features based on the results

# In[25]:


#Handle data Multicollinearity:

# compute the vif for all given features
def compute_vif(considered_features):
    
    X = X_train[considered_features]
    # the calculation of variance inflation requires a constant
    X['intercept'] = 1
    
    # create dataframe to store vif values
    vif = pd.DataFrame()
    vif["Variable"] = X.columns
    vif["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    vif = vif[vif['Variable']!='intercept']
    return vif

# features to consider removing
considered_features = ['emp.var.rate', 'cons.price.idx', 'euribor3m', 'nr.employed']


# compute vif 
compute_vif(considered_features).sort_values('VIF', ascending=False)


# In[26]:


# compute vif values after removing a feature
considered_features.remove('emp.var.rate')
compute_vif(considered_features).sort_values('VIF', ascending=False)


# In[27]:


# compute vif values after removing one more feature
considered_features.remove('euribor3m')
compute_vif(considered_features).sort_values('VIF', ascending=False)

#Now that the variance inflation factors are all within the acceptable range, the derived model will be more likely to yield statistically significant results.
#Conclusion: we need to drop two features: emp.var.rate and euribor3m before building the models;


# In[28]:


# remove two features on both train and test data set:
X_train = pd.DataFrame(X_train.drop(columns=['emp.var.rate','euribor3m'], axis = 1)) 
print(X_train.shape)
X_test = pd.DataFrame(X_test.drop(columns=['emp.var.rate','euribor3m'], axis = 1)) 
print(X_test.shape)


# ## 4.3 Detecting and treating on outliers
# * Detect and treat the outliers in variable "age,campaign" (Note that "duration" is already dropped)

# In[29]:


##Detect and treat the outliers in variable "age"
#IQR in age for train dataset
Q1 = np.percentile(X_train['age'], 25,interpolation = 'midpoint') 
Q3 = np.percentile(X_train['age'], 75,interpolation = 'midpoint')
IQR = Q3 - Q1
print(Q1)
print(Q3)
print(IQR)
# Calculate the upper and lower bound, detect the outliers
upper = Q3+1.5*IQR
print("upper bound:", upper)
lower = Q1-1.5*IQR
print("Lower bound:", lower)
outliers_train = X_train.loc[(X_train['age'] <= 9.5) | (X_train["age"] >= 69.5)]
#print(outliers_train) #We've found 384 outliers for the variable 'age' for training dataset;


# In[30]:


#Since the outliers does not have a strong association with the outcome, 
#instead of dropping them, we decided to cap them using a 99 percentile value:
# Computing 99th percentiles and replacing the outliers
ninetynine_percentile_train = np.percentile(X_train['age'],99)
print(ninetynine_percentile_train) #99% of age distribution is 71;
X_train['age'].values[X_train['age'].values > 71] = 71
#boxplot after treating the outliers
sns.boxplot(X_train['age']).set(title='Boxplot of the age after treating the outliers') #Looks great!


# In[31]:


#IQR in age for test dataset
Q1_test = np.percentile(X_test['age'], 25,interpolation = 'midpoint') 
Q3_test = np.percentile(X_test['age'], 75,interpolation = 'midpoint')
IQR_test = Q3_test - Q1_test
print(Q1_test)
print(Q3_test)
print(IQR_test)
# Calculate the upper and lower bound, detect the outliers
upper_test = Q3_test+1.5*IQR_test
print("upper bound:", upper)
lower_test = Q1_test-1.5*IQR_test
print("Lower bound:", lower)
outliers_test = X_test.loc[(X_test['age'] <= 9.5) | (X_test["age"] >= 69.5)]
#print(outliers_test) #We've found 84 outliers for the variable 'age' for test dataset;


# In[32]:


# Computing 99th percentiles and replacing the outliers
ninetynine_percentile_test = np.percentile(X_test['age'],99)
print(ninetynine_percentile_test) #99% of age distribution is 70;
X_test['age'].values[X_test['age'].values > 70] = 70
#boxplot after treating the outliers
sns.boxplot(X_test['age']).set(title='Boxplot of the age after treating the outliers') #Looks great!


# In[33]:


##Detect and treat the outliers in variable "campagin"
#IQR in age for train dataset
Q1 = np.percentile(X_train['campaign'], 25,interpolation = 'midpoint') 
Q3 = np.percentile(X_train['campaign'], 75,interpolation = 'midpoint')
IQR = Q3 - Q1
print(Q1)
print(Q3)
print(IQR)
# Calculate the upper and lower bound, detect the outliers
upper = Q3+1.5*IQR
print("upper bound:", upper)
lower = Q1-1.5*IQR
print("Lower bound:", lower)
outliers_train = X_train.loc[(X_train['campaign'] <= -2.0) | (X_train["campaign"] >= 6.0)]
#print(outliers_train) #We've found 2730 outliers for the variable 'campaign' for training dataset;


# In[34]:


# Computing 99th percentiles and replacing the outliers
ninetynine_percentile_train = np.percentile(X_train['campaign'],99)
print(ninetynine_percentile_train) #99% of campaign distribution for trainning data set is 15;
X_train['campaign'].values[X_train['campaign'].values > 15] = 15
#boxplot after treating the outliers
sns.boxplot(X_train['campaign']).set(title = 'Boxplot of the campaign after treating the outliers')
#Looks better!


# In[35]:


#IQR in campaign for test dataset
Q1_test = np.percentile(X_test['campaign'], 25,interpolation = 'midpoint') 
Q3_test = np.percentile(X_test['campaign'], 75,interpolation = 'midpoint')
IQR_test = Q3_test - Q1_test
print(Q1_test)
print(Q3_test)
print(IQR_test)
# Calculate the upper and lower bound, detect the outliers
upper = Q3_test+1.5*IQR_test
print("upper bound:", upper)
lower = Q1_test-1.5*IQR_test
print("Lower bound:", lower)
outliers_test = X_test.loc[(X_test['campaign'] <= -2.0) | (X_test["campaign"] >= 6.0)]
#print(outliers_test) #We've found 655 outliers for the variable 'campaign' for training dataset;


# In[36]:


# Computing 99th percentiles and replacing the outliers
ninetynine_percentile_test = np.percentile(X_test['campaign'],99)
print(ninetynine_percentile_test) #99% of campaign distribution for testing data set is 13.65;
# We use 14 as the replace value since campaign is an interger value.
X_test['campaign'].values[X_test['campaign'].values > 14] = 14
#boxplot after treating the outliers
sns.boxplot(X_test['campaign']).set(title=
                                     'Boxplot of the campaign after treating the outliers') 
#Outliers are much less after the treatment.


# ## 4.4 Feature Engineering on Categorical Data
# * First, encoding all categorical data by using one hot encoding;
# * Then Use stepwise selection to remove unnessary features;

# In[37]:


#Now, let's encoding all the categorical variables in trainning data set:

X_train = X_train.reset_index(drop=True)
    
def label_encoder(df,column):
    le=preprocessing.LabelEncoder()
    le.fit_transform(df[column])
    ohe = preprocessing.OneHotEncoder()
    temp_array= ohe.fit_transform(df[[column]]).toarray()
    column_names = [column+"_"+str(m) for m in le.classes_]
    return(pd.DataFrame(temp_array,columns=column_names))
     
numerical_variables = [col for col in X_train.columns 
                       if X_train[col].dtype in ['int64', 'float64','uint8','int32']] 

print(numerical_variables)

categorical_variables = [col for col in X_train.columns 
                         if X_train[col].dtype not in ['int64', 'float64','uint8','int32']] 

print(categorical_variables)

new_X_train = X_train[numerical_variables]

for column in categorical_variables:
    new_X_train = pd.concat([new_X_train,label_encoder(X_train,column)],axis=1)

new_X_train.columns


# In[38]:


print(new_X_train.shape)
print(new_X_train.dtypes)


# In[39]:


y_train = list(y_train)
# stepwise selection to remove unnessary features: (running times: about 1 minute)
def stepwise_selection(data, target,SL_in=0.05,SL_out = 0.05):
    initial_features = data.columns.tolist()
    best_features = []
    while (len(initial_features)>0):
        remaining_features = list(set(initial_features)-set(best_features))
        new_pval = pd.Series(index=remaining_features)
        for new_column in remaining_features:
            model = sm.OLS(target, sm.add_constant(data[best_features+[new_column]])).fit()
            new_pval[new_column] = model.pvalues[new_column]
        min_p_value = new_pval.min()
        if(min_p_value<SL_in):
            best_features.append(new_pval.idxmin())
            while(len(best_features)>0):
                best_features_with_constant = sm.add_constant(data[best_features])
                p_values = sm.OLS(target, best_features_with_constant).fit().pvalues[1:]
                max_p_value = p_values.max()
                if(max_p_value >= SL_out):
                    excluded_feature = p_values.idxmax()
                    best_features.remove(excluded_feature)
                else:
                    break 
        else:
            break
    return best_features
best_features = stepwise_selection(new_X_train,y_train)
print(best_features)


# * Based on the results of stepwise selection, 25 features are significant to y_train, thus we drop the result of them;

# In[40]:


#drop unnessary features in trainning data set:
selected_X_train = new_X_train.drop(columns=[col for col in new_X_train 
                                             if col not in best_features])


# In[41]:


#Based on the results on the trainning dataset, 
#let's encoding all the categorical variables and drop all necessary features in the test data set
X_test = X_test.reset_index(drop=True)

    
def label_encoder(df,column):
    le=preprocessing.LabelEncoder()
    le.fit_transform(df[column])
    ohe = preprocessing.OneHotEncoder()
    temp_array= ohe.fit_transform(df[[column]]).toarray()
    column_names = [column+"_"+str(m) for m in le.classes_]
    return(pd.DataFrame(temp_array,columns=column_names))
     
numerical_variables = [col for col in X_test.columns 
                       if X_test[col].dtype in ['int64', 'float64','uint8','int32']] 

print(numerical_variables)

categorical_variables = [col for col in X_test.columns 
                         if X_test[col].dtype not in ['int64', 'float64','uint8','int32']] 
print(categorical_variables)

new_X_test = X_test[numerical_variables]

for column in categorical_variables:
    new_X_test = pd.concat([new_X_test,label_encoder(X_test,column)],axis=1)

new_X_test.columns


# In[42]:


print(new_X_test.shape)
print(new_X_test.dtypes)
y_test = list(y_test)


# In[43]:


#drop unnessary features in test dataset:
selected_X_test = new_X_test.drop(columns=[col for col in new_X_test 
                                             if col not in best_features])
selected_X_test.head()


# ## 4.5 Use SMOTE to Balance the dataset

# In[44]:


y_test = pd.DataFrame (y_test, columns = ['y'])
print(y_test.shape)
y_train = pd.DataFrame (y_train, columns = ['y'])
print(y_train.shape)


# In[45]:


X_test = pd.DataFrame(selected_X_test)
print(X_test.shape)
X_train = pd.DataFrame(selected_X_train)
print(X_train.shape)


# In[46]:


#Lastly, let's use SMOTE to balance the dataset:
sm = SMOTE(random_state=42)
X_train_smote, y_train_smote = sm.fit_resample(X_train, y_train)
# Recheck inbalance
y_train['y'].value_counts()
X_test_smote, y_test_smote = sm.fit_resample(selected_X_test, y_test)
# Recheck inbalance
y_test['y'].value_counts()


# In[47]:


print(X_train_smote.shape)
print(y_train_smote.shape)


# In[48]:


print(X_test_smote.shape)
print(y_test_smote.shape)


# * Note that we have done all the data pre-processing job, and will start fitting models now.
# * Note that we now have trainning dataset {X_train_smote, y_train_smote} and testing dataset {X_test_somte, y_test_smote}

# # 5. Staring from Logistic Regression
# *   Fit logistic regression and visualize the ROC curve;
# *   Tune Hyperparameters by using Grid search cross-validation;
# *   Refit the model;
# *   Model Performance;
# *   Improve performance by scaling our data;

# In[52]:


#fit logistic regression
logreg = LogisticRegression()
logreg.fit(X_train_smote, y_train_smote)


# In[53]:


from sklearn.metrics import classification_report
#predicting probablities
y_pred = logreg.predict(X_test_smote)
y_pred_probs_log = logreg.predict_proba(X_test_smote)[:, 1]
print(y_pred_probs_log[0])
print(classification_report(y_test_smote,y_pred))


# In[54]:


#plotting the ROC curve
fpr,tpr, thresholds = roc_curve(y_test_smote, y_pred_probs_log)
plt.plot([0,1],[0,1],'k--')
plt.plot(fpr,tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Logistic Regression ROC Curve')
plt.show()


# In[55]:


print(roc_auc_score(y_test_smote, y_pred_probs_log))
#the auc value(area under curve) for the simplest logistic regression model is 0.80!


# In[56]:


#logistic regression default on test data set
test_score_log_default = logreg.score(X_test_smote,y_test_smote)
print(test_score_log_default)


# In[57]:


#Now, we are using grid search to tune the model

from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import GridSearchCV
import time
start_time = time.time()

# define models and parameters
X, y = (X_train_smote, y_train_smote)
model = LogisticRegression()
solvers = ['newton-cg', 'lbfgs', 'liblinear']
penalty = ['l2']
c_values = [100, 10, 1.0, 0.1, 0.01]

# define grid search
grid = dict(solver=solvers,penalty=penalty,C=c_values)
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
grid_search = GridSearchCV(estimator=model, param_grid=grid, n_jobs=-1, cv=cv, scoring='accuracy',error_score=0)
grid_result = grid_search.fit(X, y)

# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))
print("Execution time: " + str((time.time() - start_time)) + ' ms')


# In[58]:


#Evaluating on the test set
test_score_log = grid_search.score(X_test_smote,y_test_smote)
print(test_score_log) #this is even slightly better than our highest accuarcy score in training dataset


# In[59]:


# See what the best_estimator_ property is
print((grid_search.best_estimator_))


# In[60]:


# Create an array of predictions directly using the best_estimator_ property
predictions = grid_search.best_estimator_.predict(X_test_smote)

# Take a look to confirm it worked, this should be an array of 1's and 0's
print(predictions[0:5])

# classfication report for the best logistic regression model
print(classification_report(y_test_smote,predictions))


# In[61]:


# Get the ROC-AUC score
predictions_proba = grid_search.best_estimator_.predict_proba(X_test_smote)[:,1]
print("ROC-AUC Score \n", roc_auc_score(y_test_smote, predictions_proba))


# In[62]:


#scaling our data
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_smote)
X_test_scaled = scaler.fit_transform(X_test_smote)
print(np.mean(X_train_scaled), np.std(X_train_scaled))
print(X_test_scaled.shape)
print(y_test_smote)


# In[63]:


#try fitting logistic regression by using standardied data
logreg_scaled = LogisticRegression(C=100, solver='newton-cg',penalty ='l2')
logreg_scaled.fit(X_train_scaled, y_train_smote.values.ravel())


# In[64]:


#predictions and classfication report --does not see too much help
predictions_logreg_scaled = logreg_scaled.predict(X_test_scaled)
print(classification_report(y_test_smote,predictions_logreg_scaled))


# In[65]:


#Evaluating on the test set
test_score_log_scaled =logreg_scaled.score(X_test_scaled,y_test_smote)
print(test_score_log_scaled) 
#this is weaker performance than our previous model--logistic regression with best hyperparameters


# In[66]:


print(roc_auc_score(y_test_smote, predictions_logreg_scaled))


# # 6. Random Forest
# *   Fit random forest
# *   Tune Hyperparameters by using Grid search cross-validation;
# *   Refit the model;
# *   Model Performance;

# In[144]:


#fit random forest with n_estimators = 100
clf1 = RandomForestClassifier(n_estimators = 100)
clf1.fit(X_train_smote, y_train_smote.values.ravel())
y_pred_2 = clf1.predict(X_test_smote)


# In[145]:


y_pred_probs_random = clf1.predict_proba(X_test_smote)[:, 1]
print(y_pred_probs_random[0])
print(classification_report(y_test_smote,y_pred_2))


# In[146]:


#plotting the ROC curve
fpr,tpr, thresholds = roc_curve(y_test_smote, y_pred_probs_random)
plt.plot([0,1],[0,1],'k--')
plt.plot(fpr,tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Random Forest ROC Curve')
plt.show()


# In[147]:


print(roc_auc_score(y_test_smote, y_pred_probs_random))


# In[217]:


#Evaluating on the test set
test_score_random_default =clf1.score(X_test_smote,y_test_smote)
print(test_score_random_default) 


# In[112]:


# define dataset
X, y = (X_train_smote, y_train_smote.values.ravel())

# define models and parameters
model = RandomForestClassifier()
n_estimators = [10, 100, 1000]
max_features = ['sqrt', 'log2']
start_time = time.time()

# define grid search
grid = dict(n_estimators=n_estimators,max_features=max_features)
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
grid_search = GridSearchCV(estimator=model, param_grid=grid, n_jobs=-1, cv=cv, scoring='accuracy',error_score=0)
grid_result = grid_search.fit(X, y)

# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))
print("Execution time: " + str((time.time() - start_time)) + ' ms')


# In[113]:


#Evaluating on the test set
test_score_random = grid_search.score(X_test_smote,y_test_smote)
print(test_score_random) 


# In[114]:


# Create an array of predictions directly using the best_estimator_ property
predictions_2 = grid_search.best_estimator_.predict(X_test_smote)

# Take a look to confirm it worked, this should be an array of 1's and 0's
print(predictions_2[0:5])

# classfication report for the best logistic regression model
print(classification_report(y_test_smote,predictions_2))


# In[222]:


#roc-auc score for random forest with best hyperparameters
random_best = RandomForestClassifier(max_features = 'log2', n_estimators = 1000)
random_best.fit(X_train_smote, y_train_smote.values.ravel())
y_pred_random_best = random_best.predict(X_test_smote)


# In[223]:


y_pred_probs_random_best = random_best.predict_proba(X_test_smote)[:, 1]
print(y_pred_probs_random_best[0])
print(classification_report(y_test_smote,y_pred_random_best))


# In[224]:


print(roc_auc_score(y_test_smote, y_pred_probs_random_best))


# # 7. KNN
# *   Fit KNN;
# *   Tune Hyperparameters by using Grid search cross-validation;
# *   Refit the model;
# *   Model Performance;
# *   Compare with the scaled KNN model;

# In[115]:


#fit knn by using n_neighbors = 100
knn = KNeighborsClassifier(n_neighbors = 100)
knn.fit(X_train_smote,y_train_smote.values.ravel())
prediction_3 = knn.predict(X_test_smote)


# In[116]:


# Score report
print(classification_report(y_test_smote,prediction_3))


# In[117]:


y_pred_probs_knn = knn.predict_proba(X_test_smote)[:, 1]
print(y_pred_probs_knn[0])
#plotting the ROC curve
fpr,tpr, thresholds = roc_curve(y_test_smote, y_pred_probs_knn)
plt.plot([0,1],[0,1],'k--')
plt.plot(fpr,tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('KNN ROC Curve')
plt.show()


# In[118]:


print(roc_auc_score(y_test_smote, prediction_3))


# In[218]:


#Evaluating on the test set
test_score_knn_default = knn.score(X_test_smote,y_test_smote)
print(test_score_knn_default) 


# In[119]:


#Now, we are using grid search to tune the model
X, y = (X_train_smote, y_train_smote.values.ravel())

# define models and parameters
model = KNeighborsClassifier()
n_neighbors = range(1, 21, 2)
weights = ['uniform', 'distance']
metric = ['euclidean', 'manhattan', 'minkowski']
start_time = time.time()

# define grid search
grid = dict(n_neighbors=n_neighbors,weights=weights,metric=metric)
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
grid_search = GridSearchCV(estimator=model, param_grid=grid, n_jobs=-1, cv=cv, scoring='accuracy',error_score=0)
grid_result = grid_search.fit(X, y)

# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))
print("Execution time: " + str((time.time() - start_time)) + ' ms')


# In[120]:


#Evaluating on the test set
test_score_knn = grid_search.score(X_test_smote,y_test_smote)
print(test_score_knn) 


# In[121]:


# Create an array of predictions directly using the best_estimator_ property
predictions_3 = grid_search.best_estimator_.predict(X_test_smote)

# Take a look to confirm it worked, this should be an array of 1's and 0's
print(predictions_3[0:5])

# classfication report for the best logistic regression model
print(classification_report(y_test_smote,predictions_3))


# In[226]:


#try fitting knn by using best hyperparameters
knn_best = KNeighborsClassifier(metric = 'euclidean', n_neighbors = 3, weights = 'distance')
knn_best.fit(X_train_smote, y_train_smote.values.ravel())
prediction_knn_best = knn_best.predict(X_test_smote)


# In[227]:


print(roc_auc_score(y_test_smote, prediction_knn_best))


# In[126]:


#try fitting knn by using standardied data
knn_scaled = KNeighborsClassifier(metric = 'euclidean', n_neighbors = 3, weights = 'distance')
knn_scaled.fit(X_train_scaled, y_train_smote.values.ravel())
prediction_4 = knn_scaled.predict(X_test_scaled)


# In[128]:


# Score report
print(classification_report(y_test_smote,prediction_4))
print(roc_auc_score(y_test_smote, prediction_4))


# In[219]:


#Evaluating on the test set
test_score_knn_scaled = knn_scaled.score(X_test_scaled,y_test_smote)
print(test_score_knn_scaled) 


# # 8. Stochastic Gradient Boosting
# *   Fit Stochastic Gradient Boosting;
# *   Tune Hyperparameters by using Grid search cross-validation;
# *   Refit the model;
# *   Model Performance;

# In[154]:


#fitting stochastic gradient boosting
from sklearn.ensemble import GradientBoostingClassifier
clf2 = GradientBoostingClassifier(n_estimators=100)
clf2.fit(X_train_smote,y_train_smote.values.ravel())
prediction_4 = clf2.predict(X_test_smote)


# In[155]:


# Score report
print(classification_report(y_test_smote,prediction_4))


# In[156]:


y_pred_probs_stochastic = clf2.predict_proba(X_test_smote)[:, 1]
print(y_pred_probs_stochastic[0])
#plotting the ROC curve
fpr,tpr, thresholds = roc_curve(y_test_smote, y_pred_probs_stochastic)
plt.plot([0,1],[0,1],'k--')
plt.plot(fpr,tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Stochastic Gradient Boosting ROC Curve')
plt.show()


# In[157]:


print(roc_auc_score(y_test_smote, prediction_4))


# In[221]:


#Evaluating on the test set
test_score_boosting_default = clf2.score(X_test_smote,y_test_smote)
print(test_score_boosting_default) 


# In[138]:


# define dataset
X, y = (X_train_smote, y_train_smote.values.ravel())

# define models and parameters
model = GradientBoostingClassifier()
n_estimators = [10, 100, 1000]
learning_rate = [0.001, 0.01, 0.1]
subsample = [0.5, 0.7, 1.0]
max_depth = [3, 7, 9]
start_time = time.time()
# define grid search
grid = dict(learning_rate=learning_rate, n_estimators=n_estimators, subsample=subsample, max_depth=max_depth)
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
grid_search = GridSearchCV(estimator=model, param_grid=grid, n_jobs=-1, cv=cv, scoring='accuracy',error_score=0)
grid_result = grid_search.fit(X, y)

# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))
print("Execution time: " + str((time.time() - start_time)) + ' ms')


# In[204]:


#Evaluating on the test set
test_score_boosting = grid_search.score(X_test_smote,y_test_smote)
print(test_score_boosting) 


# In[228]:


#fitting models with the best hyperparameters and get the roc score
boosting_best = GradientBoostingClassifier(learning_rate = 0.01, max_depth = 9, n_estimators = 1000, subsample = 0.5)
boosting_best.fit(X_train_smote,y_train_smote.values.ravel())
prediction_5 = boosting_best.predict(X_test_smote)


# In[229]:


print(roc_auc_score(y_test_smote, prediction_5))


# # 9. Model Evaluation
# *   K fold cross validation score for each model
# *   Comparing feature importance

# In[9]:


from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
# Build the k-fold cross-validator
kfold = KFold(n_splits=3, random_state=42,shuffle=True)
#mean score for logi_1
result_logi_1 = cross_val_score(logreg,X_train_smote, y_train_smote.values.ravel(),cv=kfold, scoring='accuracy')
print(result_log_1.mean())


# In[183]:


#mean score for logi_scaled
result_log_2 = cross_val_score(logreg_scaled,X_train_scaled, y_train_smote.values.ravel(),cv=kfold, scoring='accuracy')
print(result_log_2.mean())


# In[159]:


#mean score for random forest default
result_random_1 = cross_val_score(clf1,X_train_smote,y_train_smote.values.ravel(),cv=kfold, scoring='accuracy')
print(result_random_1.mean())


# In[160]:


#mean score for knn default
result_knn_1 = cross_val_score(knn,X_train_smote,y_train_smote.values.ravel(),cv=kfold, scoring='accuracy')
print(result_knn_1.mean())


# In[162]:


#mean score for knn scaled
result_knn_scaled = cross_val_score(knn_scaled,X_train_scaled, y_train_smote.values.ravel(),cv=kfold, scoring='accuracy')
print(result_knn_scaled.mean())


# In[161]:


#mean score for Stochastic Gradient Boosting default
result_stochastic = cross_val_score(clf2,X_train_smote,y_train_smote.values.ravel(),cv=kfold, scoring='accuracy')
print(result_stochastic.mean())


# In[188]:


#Model Comparsion: default model performance with unscaled data
models = {"Logistic Regression": LogisticRegression(),"KNN": KNeighborsClassifier(n_neighbors = 100),
          "Random Forest":  RandomForestClassifier(n_estimators = 100), 
          "Stochastic Gradient Boosting": GradientBoostingClassifier(n_estimators=100)}
results = []
for model in models.values():
    kf = KFold(n_splits=6, random_state=42, shuffle=True)
    cv_results = cross_val_score(model, X_train_smote,y_train_smote.values.ravel(), 
                                 cv=kf, scoring = 'roc_auc')
    results.append(cv_results)
plt.boxplot(results, labels=models.keys())
plt.xticks(rotation=30)
plt.show()


# In[189]:


#Model Comparsion: default model performance with scaled data
models = {"Logistic Regression": LogisticRegression(),"KNN": KNeighborsClassifier(n_neighbors = 100),
          "Random Forest":  RandomForestClassifier(n_estimators = 100), 
          "Stochastic Gradient Boosting": GradientBoostingClassifier(n_estimators=100)}
results = []
for model in models.values():
    kf = KFold(n_splits=6, random_state=42, shuffle=True)
    cv_results = cross_val_score(model, X_train_scaled,y_train_smote.values.ravel(), 
                                 cv=kf,scoring = 'roc_auc')
    results.append(cv_results)
plt.boxplot(results, labels=models.keys())
plt.xticks(rotation=30)
plt.show()
print(results)


# In[220]:


#Model Comparsion: model performance with best hyperparameters with unscaled data
models = {"Logistic Regression": LogisticRegression(C=100, solver='newton-cg',penalty ='l2'),
          "KNN": KNeighborsClassifier(metric = 'euclidean', n_neighbors = 3, weights = 'distance'),
          "Random Forest":  RandomForestClassifier(max_features = 'log2', n_estimators = 1000), 
          "Stochastic Gradient Boosting": GradientBoostingClassifier(learning_rate = 0.01, 
                                                                     max_depth = 9, 
                                                                     n_estimators = 1000, 
                                                                     subsample = 0.5)}
results = []
for model in models.values():
    kf = KFold(n_splits=6, random_state=42, shuffle=True)
    cv_results = cross_val_score(model, X_train_smote,y_train_smote.values.ravel(), 
                                 cv=kf, scoring = 'roc_auc')
    results.append(cv_results)
plt.boxplot(results, labels=models.keys())
plt.xticks(rotation=30)
plt.show()
print(mean(results))


# In[191]:


#Model Comparsion: model performance with best hyperparameters with scaled data
models = {"Logistic Regression": LogisticRegression(C=100, solver='newton-cg',penalty ='l2'),
          "KNN": KNeighborsClassifier(metric = 'euclidean', n_neighbors = 3, weights = 'distance'),
          "Random Forest":  RandomForestClassifier(max_features = 'log2', n_estimators = 1000), 
          "Stochastic Gradient Boosting": GradientBoostingClassifier(learning_rate = 0.01, 
                                                                     max_depth = 9, 
                                                                     n_estimators = 1000, 
                                                                     subsample = 0.5)}
results = []
for model in models.values():
    kf = KFold(n_splits=6, random_state=42, shuffle=True)
    cv_results = cross_val_score(model, X_train_scaled,y_train_smote.values.ravel(), 
                                 cv=kf, scoring = 'roc_auc')
    results.append(cv_results)
plt.boxplot(results, labels=models.keys())
plt.xticks(rotation=30)
plt.show()


# In[194]:


#extract feature importance from random forest
from sklearn.inspection import permutation_importance
rf = RandomForestClassifier(max_features = 'log2', n_estimators = 1000)
rf.fit(X_train_smote,y_train_smote.values.ravel())
rf.feature_importances_


# In[203]:


# Visualizing Important Features in Random Forest
importance = rf.feature_importances_
sortedImp = pd.Series(rf.feature_importances_, index=X_train_smote.columns).sort_values(ascending = True)

# plot feature importance
plt.figure(figsize = (10,10))
plt.barh(sortedImp.index, sortedImp)
plt.xlabel("Feature")
plt.ylabel("Feature Importance Score")
plt.show()

