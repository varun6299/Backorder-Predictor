#!/usr/bin/env python
# coding: utf-8

# ## About the Topic
# A backorder is an item that is not in stock at the moment. The customer can still order it, but delivery will only happen once the stock has been replenished. This is different from a situation when a product is labeled as being “out-of-stock” on the web platform which means the customer can’t order the product. However, backorders can still be placed despite the product not being in stock at a given moment in time. However, this causes a lot of additional inventory costs and wait for the customers, which is undesirable.
#
# ### Objective
# The aim is to create a prediction model largely to identify whether items could be going out of stock or on backorder. This will help us to keep adequate or surplus stock of those products in order to meet demands of customers and not lose sales opportunities.
#
# ### Data Description
# The list of all features and explanation of certain features (only unknown or technical jargons) in the features given the dataset have been provided below :
#
# - **sku** : a stock keeping unit is a distinct type of item for sale, such as a product or service, and all attributes associated with the item type that distinguish it from other item types
#
# - **national_inv**: The present national level of inventory of the product
#
# - **lead_time** :  lead time in inventory management is the amount of time between when a purchase order is placed to replenish products and when the order is received in the warehouse. Order lead times can vary between suppliers; the more suppliers involved in the chain, the longer the lead time is likely to be.
#
# - **in_transit_qty** : In transit inventory, also called transportation inventory or goods in transit, is any good shipped by a seller but not yet received by a buyer
#
# - **forecast_3_month** : Forecasted sales of the product for the next 3 months.
#
# - **forecast_6_month** : Forecasted sales of the product for the next 6 months.
#
# - **forecast_9_month** : Forecasted sales of the product for the next 9 months.
#
# - **sales_1_month** : Actual Sales of the product in the last 1 month.
#
# - **sales_3_month** : Actual Sales of the product in the last 3 months.
#
# - **sales_6_month** : Actual Sales of the product in the last 6 months.
#
# - **sales_9_month** : Actual Sales of the product in the last 9 months.
#
# - **min_bank**: Minimum amount of stock recommended to have.
#
# - **potential_issue**: Any problem identified with the product or part.
#
# - **pieces_past_due** : Assumption - The stock items that have been kept in stock for long time, past their expected life span.
#
# - **perf_6_month_avg** : Average performance of product compared to expected performance in terms of units sold over last 6 months. [NOTE - true meaning of the feature is not clear or available. Assumption made based on values].
#
# - **perf_12_month_avg**: Average performance of product compared to expected performance in terms of units sold over last 12 months. [NOTE - true meaning of the feature is not clear or available. Assumption made based on values].
#
# - **local_bo_qty** : BO QTY in inventory managment refers to the Back Order Quantity. To find Back Order Quantity, divide the number of undeliverable orders by the total number of orders and multiply the result by 100.
#
# - **deck_risk** : Assumption - It is the risk associated with keeping the items in stock and could relate to damage, shelf life, theft, etc.
# - oe_constraint
# - **ppap_risk** : Short for Production Part Approval Process, it is a risk reduction process which is used as a risk classification and qualification process which is used to determine whether a production run will produce parts with consistency and repeatability. It is done prior to product release
# - **stop_auto_buy** : Has the auto buy for the product, which was back ordered, cancelled.
# - rev_stop
#
# **TARGET FEATURE** : went_on_backorder - Whether an items was backordered or not

# ## NOTE -
#
# - The following codes were written in 3 different Amazon Sagemaker files and the final 3 notebooks were appended into 1.
# - Some additional codes written like importing packages and dataset description at the start of each file have been removed for presentation purposes

# ### Initializing Logger

# In[1]:


import logging

logging.basicConfig(filename = "Logging.txt",level=logging.INFO,filemode='a',format = '%(asctime)s %(levelname)s-%(message)s',datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger()


# ### Importing the Pre-requisite Packages and Dataset

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as exp
import warnings
warnings.filterwarnings('ignore')
logger.info("All pre-requisite packages imported")


data = pd.read_csv("Kaggle_Training_Dataset_v2.csv")
logger.info("Dataset Imported successfully")


## Data Understanding

### Seeing the first five records
data.head()

### Checking the names of the columns
data.columns

### Checking data types of the variables
data.info()

### Checking the names of the numerical variables
data.select_dtypes(include=np.number).columns


### Checking the number of numerical variables
len(data.select_dtypes(include=np.number).columns)


### Checking the first 5 records for only numerical fields.
data.select_dtypes(include=np.number).head()


### Checking the names of the categorical variables
data.select_dtypes(exclude=np.number).columns

### Checking the number of categorical variables
len(data.select_dtypes(exclude=np.number).columns)

### Checking the first 5 records for only numerical fields.
data.select_dtypes(exclude=np.number).head()


### Counting unique elements in each field
data.nunique()


data.nunique() / len(data) * 100

### Checking the missing values total & % totalin each field
data.isnull().sum()

data.isnull().sum() / len(data) * 100


### Checking target class proportion / check for imbalance
plt.figure(figsize=(5,6))
proportions = data.went_on_backorder.value_counts(1)*100
labels = data.went_on_backorder.value_counts(1).index
plt.pie(x=proportions,labels=labels,autopct="%.2f")
plt.title("Proportion of Target Classes")
plt.show()


### Observations -
# - In total, there are 23 variables, of which 15 are numerical and 8 are categorical
# - All the numerical features are continous in nature.
# - sku is unique for each record as it helps to identify each item uniquely.
# - In terms of missing values, we can see that in all variables except sku, there is 1 missing value. This record can be dropped.
# - Only lead time has more than 1 missing values. It has close to 6% missing values.
# - Also, based on the proportion of target classes, with more than 99% of data being for "No" class, we can say that our data is heavily imbalanced.

## Data Cleaning
logger.info("Data Cleaning Started")


### Checking for trailing spaces in each feature
for i in data.columns:
    if i == i.strip():
        print(f"{i} has no trailing space","",sep="\n")
    else:
        print(f"{i} has trailing space","",sep="\n")
logger.info("Trailing spaces in column names checked")



### This is the record with values for all variables missing. Hence it will be dropped
data.drop("sku",axis=1)[(data.drop("sku",axis=1).isnull()).all(axis=1)].index


data.drop(1687860,inplace=True)
logger.info("Records no. 1687860 being dropped due to being total empty")


### Checking unique values in each feature.
for i in data.select_dtypes(include=np.object).columns[1:]:
    print(i,":",data[i].unique())
logger.info("Unique values in categorical features checked")


### Observations -
# - None of the columns names have any trailing space.
# - All the categorical features have 2 unique values i.e Yes and No, which look alright.
# - The one records with all values missing has been dropped.

logger.info("Data Cleaning complete")
