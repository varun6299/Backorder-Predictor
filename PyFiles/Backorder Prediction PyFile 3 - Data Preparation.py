# Data Preparation

X = pd.read_csv("Xsample.csv")
y = pd.read_csv("Ysample.csv")
y.replace({"No":0,"Yes":1},inplace=True)


# checking shape of Input features
X.shape


# checking shape of target feature
y.shape


# Removing unwanted columns
X.drop(["Unnamed: 0","sku"],axis=1,inplace=True)
y.drop("Unnamed: 0",axis=1,inplace=True)
logger.info("Dropped unnamed column created and the stock keeping unit feature")


# Importing libarires
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder, OrdinalEncoder,PowerTransformer
from sklearn.model_selection import train_test_split


### NOTE -
#
# - We noted at the end of EDA that the imbalance in the target classes needs to be corrected. Due to a very high ratio difference between  backorders and non backorders , we have decided to undersample the non backorders
#
# - However, since this project is going to be deployed on a large scale basis, we need to take care of the data leakage issue to maintain model generalizability and performance across train, test and unseen data passed by user.
#
# - For this we will first split the data into train and test samples, perform required data preparation across the samples and combine them and then undersample. We do this so that we dont apply the fit methods on the entire data and apply fit on the train and transform the test samples.
#
# - Following undersample, we will again split into train and test for model building.

### Splitting the data into train and test for preprocessing

logger.info("Splitting Data into train and test samples")
Xtrain,Xtest,ytrain,ytest = train_test_split(X,y,test_size=0.30,random_state=0,stratify=y)


### Treating the Missing Values

Xtrain.lead_time.isnull().sum() / len(Xtrain) * 100

Xtrain.lead_time.value_counts()


# Inferences about Missing values -
#
# - As observed during EDA, only 1 feature with missing values i.e lead time.
# - The % of missing values is around 6%.
# - The feature has many unique values, with around 50% being for 8 lead days.
# - We can use **simple imputer and replace values with most frequent** lead time, which is clearly 8 days.

SI = SimpleImputer(strategy="most_frequent").fit(Xtrain.lead_time.values.reshape(-1,1))
Xtrain.lead_time = SI.transform(Xtrain.lead_time.values.reshape(-1,1))
Xtest.lead_time = SI.transform(Xtest.lead_time.values.reshape(-1,1))

logger.info("Missing Values In lead time replaced successfully")

print("Missing values in training data for lead time:",Xtrain.lead_time.isnull().sum())
print("Missing values in testingdata for lead time:",Xtest.lead_time.isnull().sum())


print("Number of records in train:",len(Xtrain))
print("Number of records in test:",len(Xtest))
print("Total records in train test combined:",len(X))


### Outlier Treatment

# Checking first few records
Xtrain.head()


# Checking distribution and skewness of numerical features in xtrain and xtest

plt.figure(figsize=(15,20))
for index,col in enumerate(Xtrain.select_dtypes(include=np.number).columns):
    plt.subplot(8,2,index+1)
    plt.tight_layout(pad=2,h_pad=2)
    sns.boxplot(Xtrain[col])


print("Skewness in the numerical values of Xtrain")
Xtrain.skew()



print("Skewness in the numerical values of Xtest")
Xtest.skew()


# Inferences About Outliers -
# - We can observe that there are outliers in all numerical features.
# - Except performance of 6 month average, all other features are heavy positive skewness.
# - We can observe very extremely high outliers in all features.
# - The triple digit skewness values indicate the outliers are extreme.
# - We cannot cap such outliers due to being very far from whisker.
# - We cannot also drop the values as they are many in number.
# - Hence, we will use **power transformation to reduce the impact of outliers** since it can deal with both positive, negative and 0 value terms.

PT = PowerTransformer().fit(Xtrain.select_dtypes(include=np.number))

Xtrain[Xtrain.select_dtypes(include=np.number).columns] = PT.transform(Xtrain.select_dtypes(include=np.number))
Xtest[Xtest.select_dtypes(include=np.number).columns] = PT.transform(Xtest.select_dtypes(include=np.number))


logger.info("Numerical Features successfully transformed using Power transformation to treat the impact of extreme outliers")


# Checking distribution and skewness of numerical features in xtrain and xtest after transformations
plt.figure(figsize=(15,20))
for index,col in enumerate(Xtrain.select_dtypes(include=np.number).columns):
    plt.subplot(8,2,index+1)
    plt.tight_layout(pad=2,h_pad=2)
    sns.boxplot(Xtrain[col])


print("Skewness in the numerical values of Xtrain after transformation")
Xtrain.skew()


print("Skewness in the numerical values of Xtest after transformation")
Xtest.skew()


print("Number of records in train:",len(Xtrain))
print("Number of records in test:",len(Xtest))
print("Total records in train test combined:",len(X))


### Encoding

#  Checking count of categories in categorical features
Xtrain.select_dtypes(include=np.object).nunique()


# Inferences -
#
# - Based on the analysis of the number of categories in each feature, we observe that all categorical features have only 2 categories.
# - Due to less no. of unique categories, we can easily use **one hot encoding**

OHE = OneHotEncoder(drop="first").fit(Xtrain.select_dtypes(include=np.object))

Xtrain_OHE = pd.DataFrame(OHE.transform(Xtrain.select_dtypes(include=np.object)).toarray(),columns=OHE.get_feature_names(),index=Xtrain.index)
Xtest_OHE = pd.DataFrame(OHE.transform(Xtest.select_dtypes(include=np.object)).toarray(),columns=OHE.get_feature_names(),index=Xtest.index)


# Checking xtrain and xtest after encoding and merging ohe dataframes with respective dataframe
Xtrain.head()

Xtrain_OHE.head()

Xtrain = Xtrain.merge(Xtrain_OHE,left_index=True,right_index=True)
Xtest = Xtest.merge(Xtest_OHE,left_index=True,right_index=True)

print("Number of records in train:",len(Xtrain))
print("Number of records in test:",len(Xtest))
print("Total records in train test combined:",len(X))

Xtrain.columns


# Checking xtrain and xtest after encoding and merging
Xtrain.head()

ytrain.head()


# Dropping duplicate fields
Xtrain.drop(["ppap_risk","stop_auto_buy","deck_risk","potential_issue","oe_constraint","rev_stop"],axis=1,inplace=True)
Xtest.drop(["ppap_risk","stop_auto_buy","deck_risk","potential_issue","oe_constraint","rev_stop"],axis=1,inplace=True)

logger.info("Categorical Features successfully one hot encoded")


### Undersampling the Majority Class i.e No Backorders

#### Note -
# - Since the majority and minority classes have a drastic ratio of close to approx 99 to 1, we need to undersampe the majority classes in order to improve performance of model.
# - However, in order to retain a little bit of the real life scenario in terms of balance of backorders and non backorders, we will maintain a ratio between the two in such as way that the ratio is around 25:75 or 1:3.
# - To undersample, we will use **NearMiss technique**.


nums = ['national_inv', 'lead_time', 'in_transit_qty', 'forecast_6_month',
       'sales_6_month', 'min_bank', 'pieces_past_due', 'perf_6_month_avg',
       'local_bo_qty']


X = Xtrain.append(Xtest)
y = ytrain.append(ytest)
logger.info("Combined processed train and test samples into 1 bigger sample for undersampling")


# Checks on dataframe
Xtrain.head()


X.head()


X.tail()


Xtest.tail()


print("Number of records in train:",len(Xtrain))
print("Number of records in test:",len(Xtest))
print("Total records in train test combined:",len(X))


# Scaling numerical features

SC = StandardScaler().fit(X[list(nums)])
X[list(nums)] = SC.transform(X[list(nums)])
logger.info("Numerical features scaled for the purpose of undersampling")


## undersampling

Xtrain.shape

Xtest.shape

get_ipython().system(' pip install imblearn')


from sklearn.model_selection import train_test_split
from imblearn.under_sampling import NearMiss
from collections import Counter

NM = NearMiss(sampling_strategy=0.25)
X_under , y_under = NM.fit_resample(X,y)
logger.info("Majority Class undersampled to make it 3 times the count of minority class instances")


print(f"befor Undersampling: {y.value_counts().to_dict()}")
print(f"After Undersampling using NearMiss technique {y_under.value_counts().to_dict()}")


X_under[list(nums)] = SC.inverse_transform(X_under[list(nums)])


X_under.head()


#Saving undersampled dataset for future purposes
X_under.to_csv("X_undersampled.csv")
y_under.to_csv("Y_undersampled.csv")


## Splitting Into Train and Test for model building

Xtrain,Xtest,ytrain,ytest = train_test_split(X_under,y_under,test_size=0.30,random_state=0,stratify=y_under)
logger.info("Data Split into train and test samples for model building")


# Saving undersampled train test samples for model building
Xtrain.to_csv("xtrain.csv")
Xtest.to_csv("xtest.csv")
ytrain.to_csv("ytrain.csv")
ytest.to_csv("ytest.csv")
logger.info("train and test samples saved for model building in script 3.")
