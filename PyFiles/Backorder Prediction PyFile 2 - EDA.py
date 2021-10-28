
## Exploratory Data Analysis

### Univariate Analysis

logger.info("Exploratory Data Analysis Started")
logger.info("Part 1: Univariate Analysis")

# Checking value counts
plt.figure(figsize=(12,8))
data.select_dtypes(include=np.number).nunique().plot(kind="barh")
plt.title("Unique Value Counts",size=20)
plt.xlabel("Count")
plt.show()


# Checking missing values
plt.figure(figsize=(12,8))
data.isnull().sum().plot(kind="barh")
plt.title("Missing Value Counts",size=20)
plt.xlabel("Count")
plt.show()


# Checking distributions of numerical features
plt.figure(figsize=(15,20))
for index,col in enumerate(data.select_dtypes(include=np.number).columns):
    plt.subplot(8,2,index+1)
    plt.tight_layout(pad=2,h_pad=2)
    sns.distplot(data[col])


# Checking skewness values

data.skew()


# Checking proportion of categories in categorical variables
plt.figure(figsize=(15,12))
for index,col in enumerate(data.select_dtypes(exclude=np.number).columns[1:-1]):
    plt.subplot(3,2,index+1)
    plt.tight_layout(pad=2,h_pad=2)
    proportions = data[col].value_counts(1)*100
    labels = data[col].value_counts(1).index
    plt.pie(x=proportions,labels=labels,autopct="%.2f")
    plt.title(f"Proportion of {col}")


#### Observations -
#
# - Within numerical features, majority of columns are have large no. of unique values, indicating large amount of variance, which will be good for analysis and models.
# - In terms of missing values, only lead time has missing values, which total upto around 100K.
# - In terms of distributions of the numerical features, all the features have huge amount of positive skewness, indicated by long extended tails to right of distribution plots and also by 3 digit skewness values.
# - We observe that out of 6 categorical features, 4 features i.e potential risk, oe constraint, rev stop and stop auto buy have distribution of categories which is very skewed i.e % of "No" is more than 95%.
# - This distribution in cateogories is due to the imbalance in the dataset.

### Bi-Variate Analysis
logger.info("Part 2: Bi-variate Analysis")

# Checking relation between target and categorical features
plt.figure(figsize=(15,20))
for index,col in enumerate(data.select_dtypes(include=np.number).columns):
    plt.subplot(8,2,index+1)
    plt.tight_layout(pad=2,h_pad=2)
    sns.boxplot(y=data[col],x=data.went_on_backorder)


# Checking relation between target and categorical feature based on medians
for col in data.select_dtypes(include=np.number).columns:
    print(col)
    print(data.groupby("went_on_backorder")[col].median())
    print()


import scipy.stats as stats


# Checking statistical relation between target and numerical features using median
for col in data.select_dtypes(include=np.number).columns:
    print(col)
    print(stats.mannwhitneyu(data[data.went_on_backorder=="Yes"][col],data[data.went_on_backorder=="No"][col])[1])
    print()



# Checking statistical relation between target and numerical features using mean
for col in data.select_dtypes(include=np.number).columns:
    print(col)
    print(stats.ttest_ind(data[data.went_on_backorder=="Yes"][col],data[data.went_on_backorder=="No"][col])[1])
    print()



# Checking relation between target and categorical features
plt.figure(figsize=(15,12))
for index,col in enumerate(data.select_dtypes(exclude=np.number).columns[1:-1]):
    plt.subplot(3,2,index+1)
    plt.tight_layout(pad=2,h_pad=2)
    sns.countplot(x=data[col],hue=data.went_on_backorder)
    plt.title(f"Breakup of Backorders based on {col}",size=12)

# Checking relation between target and categorical features using cross table
pd.crosstab(index=data.went_on_backorder,columns=data.potential_issue,normalize=True,margins=True) * 100


# Checking relation between target and categorical features using cross table
pd.crosstab(index=data.went_on_backorder,columns=data.deck_risk,normalize=True,margins=True) * 100

# Checking relation between target and categorical features using cross table
pd.crosstab(index=data.went_on_backorder,columns=data.oe_constraint,normalize=True,margins=True) * 100

# Checking relation between target and categorical features using cross table
pd.crosstab(index=data.went_on_backorder,columns=data.ppap_risk,normalize=True,margins=True) * 100

# Checking relation between target and categorical features using cross table
pd.crosstab(index=data.went_on_backorder,columns=data.stop_auto_buy,normalize=True,margins=True) * 100

# Checking relation between target and categorical features using cross table
pd.crosstab(index=data.went_on_backorder,columns=data.rev_stop,normalize=True,margins=True) * 100

# Checking relation between target and categorical features using cross table
stats.chi2_contingency(pd.crosstab(data.went_on_backorder,data["deck_risk"]))[1]

# Checking statistical relation between target and categorical features using chi square
for col in data.select_dtypes(exclude=np.number).columns[1:-1]:
    print(col)
    print("p value of chi square:",stats.chi2_contingency(pd.crosstab(data.went_on_backorder,data[col]))[1])
    print()



# Checking relation between lead time and features to treat missing values in lead time
plt.figure(figsize=(20,12))
plt.subplot(1,2,1)
(data[data.went_on_backorder=="No"].lead_time.value_counts(1).sort_values(ascending=True)).plot(kind="barh",color="blue",alpha=0.5)
plt.ylabel("Days")
plt.title("Lead Time for no backorders",size=15)

# Checking relation between lead time and features to treat missing values in lead time
plt.subplot(1,2,2)
(data[data.went_on_backorder=="Yes"].lead_time.value_counts(1).sort_values(ascending=True)).plot(kind="barh",color="red",alpha=0.5)
plt.ylabel("Days")
plt.title("Lead Time for backorders",size=15)

# Checking relation between lead time and features to treat missing values in lead time
plt.figure(figsize=(15,15))
for index,i in enumerate(data.select_dtypes(include=np.object).columns[1:-1]):
    plt.subplot(3,2,index+1)
    plt.tight_layout(pad=3,h_pad=3)
    sns.violinplot(x=data[i],y=data.lead_time)
    plt.title(f"Relationship b/w leadtime and {i}")

# Checking relation between lead time and features to treat missing values in lead time
for index,i in enumerate(data.select_dtypes(include=np.object).columns[1:-1]):
    print(data.groupby(i)["lead_time"].median())
    print()

# Checking relation between lead time and features to treat missing values in lead time
plt.figure(figsize=(15,30))
for index,i in enumerate(data.select_dtypes(include=np.number).columns):
    plt.subplot(8,2,index+1)
    plt.tight_layout(pad=3,h_pad=3)
    sns.scatterplot(x=data[i],y=data.lead_time)
    plt.title(f"Relationship b/w leadtime and {i}")


# #### Observations -
#
# - In terms of the numerical features, based on analyzing relationship between independent features and predicted feature i.e went on backorder, we observed that majority of features show a very significant pattern and help in seperating back orders from none backorders.
# - Since the numerical features are heavily skewed, we use medians for comparisons.
# - Using boxplots, except min_bank, pieces_past_due, perf_6_month_avg, perf_12_month_avg and local_bo_qty, all other numerical features show clear relations with backorders/none backorders in terms of difference in distributions.
# - except min_bank, pieces_past_due, perf_6_month_avg, perf_12_month_avg and local_bo_qty don't appearently have any significant difference in medians based on boxplots.
# - However, on conducting a mann whitney u test to confirm significant difference in backorders and not backorders based on numerical features, we observed all there are significant difference between medians of the 2 target classes using any or all numerical features.
#
#
# - In terms of categorical features,it is very difficult to visually observe any pattern due to heavy imbalance in target classes.
# - However, using the chi square test of independence, we can conclude that there is a dependence/ relationship between the categorical features and the target classes of the predicted feature i.e went on backorders.
#
#
# - On trying to check which feature is significantly related to lead time in order to fill missing values in lead time, we couldnt find any such feature across categorical or numerical features as majority of the features don't have any significant relation with lead time.
#

### Multivariate Analysis

logger.info("Multivariate Analysis Started")


# Correlation heatmap
plt.figure(figsize=(12,10))
sns.heatmap(round(data.corr(),2),annot=True)
plt.title("Correlation Heatmap",size=15)
plt.show()



from statsmodels.stats.outliers_influence import variance_inflation_factor


#Checking for multicollinearity
num_cols = data.select_dtypes(include=np.number).drop("lead_time",axis=1)
vif = [variance_inflation_factor(num_cols.values,i) for i in range(len(num_cols.columns))]
VIF = pd.DataFrame({"Feature":num_cols.columns,"vif":vif})
VIF.sort_values("vif",ascending=False,inplace=True)
VIF


#### Actual Sales vs Average performance of sales



plt.figure(figsize=(15,6))
plt.subplot(1,2,1)
sns.scatterplot(y=data[data.went_on_backorder=="No"].perf_6_month_avg,x=data[data.went_on_backorder=="No"].sales_6_month,alpha=0.5,color="blue")
plt.title("Average Performance vs Sales for 6 months: No Backorders")

plt.subplot(1,2,2)
sns.scatterplot(y=data[data.went_on_backorder=="Yes"].perf_6_month_avg,x=data[data.went_on_backorder=="Yes"].sales_6_month,alpha=0.5,color="red")
plt.title("Average performance vs Sales for 6 months: Backorders")

plt.show()


#### Did Sales for a period of n month overtake forecasted sales for double the period of n months

plt.figure(figsize=(15,6))
plt.subplot(1,2,1)
sns.scatterplot(data[data.went_on_backorder=="No"].national_inv,data[data.went_on_backorder=="No"].pieces_past_due,alpha=0.5,color="blue")
plt.title("national inventory vs pieces past due: Non Backorders")

plt.subplot(1,2,2)
sns.scatterplot(data[data.went_on_backorder=="Yes"].national_inv,data[data.went_on_backorder=="Yes"].pieces_past_due,alpha=0.5,color="red")
plt.title("national inventory vs pieces past due: Backorders")
plt.show()


print("% of non backorders where national inventory of product was less than the stock which was past its shelf lifespan:",round(len(data[(data.went_on_backorder=="No")&(data.national_inv < data.pieces_past_due)]) / len(data[data.went_on_backorder=="No"]),4) * 100)
print("% of backorders where national inventory of product was less than the stock which was past its shelf lifespan:",round(len(data[(data.went_on_backorder=="Yes")&(data.national_inv < data.pieces_past_due)]) / len(data[data.went_on_backorder=="Yes"]),4) * 100)


#### Analysis of Inventory and Quantity

plt.figure(figsize=(15,6))
plt.subplot(1,2,1)
sns.scatterplot(data[data.went_on_backorder=="No"].in_transit_qty,data[data.went_on_backorder=="No"].min_bank,alpha=0.5,color="blue")
plt.title("min back vs In transit qty: Non Backorders")

plt.subplot(1,2,2)
sns.scatterplot(data[data.went_on_backorder=="Yes"].in_transit_qty,data[data.went_on_backorder=="Yes"].min_bank,alpha=0.5,color="red")
plt.title("min back vs In transit qty: Backorders")

plt.show()


print("% of non backorders where minimum stock recommended to hold was less than what was in transit:",round(len(data[(data.went_on_backorder=="No")&(data.in_transit_qty < data.min_bank)]) / len(data[data.went_on_backorder=="No"]),4) * 100)
print("% of backorders minimum stock recommended to hold was less than what was in transit:",round(len(data[(data.went_on_backorder=="Yes")&(data.in_transit_qty < data.min_bank)]) / len(data[data.went_on_backorder=="Yes"]),4) * 100)





plt.figure(figsize=(15,6))
plt.subplot(1,2,1)
sns.scatterplot(data[data.went_on_backorder=="No"].in_transit_qty,data[data.went_on_backorder=="No"].national_inv,alpha=0.5,color="blue")
plt.title("national inventory vs In transit qty: No Backorders")

plt.subplot(1,2,2)
sns.scatterplot(data[data.went_on_backorder=="Yes"].in_transit_qty,data[data.went_on_backorder=="Yes"].national_inv,alpha=0.5,color="red")
plt.title("national inventory vs In transit qty: Backorders")

plt.show()


print("% of non backorders where national level inventory of product was less than what was in transit:",round(len(data[(data.went_on_backorder=="No")&(data.national_inv < data.in_transit_qty)]) / len(data[data.went_on_backorder=="No"]),4) * 100)
print("% of backorders national level inventory of product was less than what was in transit:",round(len(data[(data.went_on_backorder=="Yes")&(data.national_inv < data.in_transit_qty)]) / len(data[data.went_on_backorder=="Yes"]),4) * 100)






plt.figure(figsize=(15,6))
plt.subplot(1,2,1)
sns.scatterplot(data[data.went_on_backorder=="No"].national_inv,data[data.went_on_backorder=="No"].min_bank,alpha=0.5,color="blue")
plt.title("min back vs national inventory: No Backorders")

plt.subplot(1,2,2)
sns.scatterplot(data[data.went_on_backorder=="Yes"].national_inv,data[data.went_on_backorder=="Yes"].min_bank,alpha=0.5,color="red")
plt.title("min back vs national inventory: Backorders")

plt.show()

print("% of non backorders where national level inventory of product was less than minimum stock that was recommended:",round((len(data[(data.went_on_backorder=="No")&(data.national_inv < data.min_bank)]) / len(data[data.went_on_backorder=="No"])) * 100,2))
print("% of backorders national level inventory of product was less than minimum stock that was recommended:",round((len(data[(data.went_on_backorder=="Yes")&(data.national_inv < data.min_bank)]) / len(data[data.went_on_backorder=="Yes"])) * 100,2))






plt.figure(figsize=(15,5))

plt.subplot(1,2,1)
sns.countplot(x=data[data.went_on_backorder=="No"].potential_issue,hue=data[data.went_on_backorder=="No"].ppap_risk)
plt.title("Relation between ppap risk and potential issues\nFor Non Backordered Products")

plt.subplot(1,2,2)
sns.countplot(x=data[data.went_on_backorder=="Yes"].potential_issue,hue=data[data.went_on_backorder=="Yes"].ppap_risk)
plt.title("Relation between ppap risk and potential issues\nFor-Backordered Products")

print("% of Backorders where there was a potential issue and ppap risk",len(data[(data.went_on_backorder=="Yes")&(data.potential_issue=="Yes")&(data.ppap_risk =="Yes")])/len(data[data.went_on_backorder=="Yes"])*100)
print("% of non Backorders where there was a potential issue and ppap risk",len(data[(data.went_on_backorder=="No")&(data.potential_issue=="Yes")&(data.ppap_risk =="Yes")])/len(data[data.went_on_backorder=="No"])*100)







plt.figure(figsize=(15,5))

plt.subplot(1,2,1)
sns.countplot(x=data[data.went_on_backorder=="No"].potential_issue,hue=data[data.went_on_backorder=="No"].deck_risk)
plt.title("Relation between deck risk and potential issues\nFor Non Backordered Products")

plt.subplot(1,2,2)
sns.countplot(x=data[data.went_on_backorder=="Yes"].potential_issue,hue=data[data.went_on_backorder=="Yes"].deck_risk)
plt.title("Relation between deck risk and potential issues\nFor Backordered Products")

plt.show()

print("% of Backorders where there was a potential issue and deck risk",len(data[(data.went_on_backorder=="Yes")&(data.potential_issue=="Yes")&(data.deck_risk =="Yes")])/len(data[data.went_on_backorder=="Yes"])*100)
print("% of non Backorders where there was a potential issue and deck risk",len(data[(data.went_on_backorder=="No")&(data.potential_issue=="Yes")&(data.deck_risk =="Yes")])/len(data[data.went_on_backorder=="No"])*100)






plt.figure(figsize=(15,5))

plt.subplot(1,2,1)
sns.countplot(x=data[data.went_on_backorder=="No"].potential_issue,hue=data[data.went_on_backorder=="No"].oe_constraint)
plt.title("Relation between oe constraint and potential issues\nFor Non Backordered Products")

plt.subplot(1,2,2)
sns.countplot(x=data[data.went_on_backorder=="Yes"].potential_issue,hue=data[data.went_on_backorder=="Yes"].oe_constraint)
plt.title("Relation between oe constraint and potential issues\nFor Backordered Products")

plt.show()

print("% of Backorders where there was a potential issue and oe constraint",len(data[(data.went_on_backorder=="Yes")&(data.potential_issue=="Yes")&(data.oe_constraint =="Yes")])/len(data[data.went_on_backorder=="Yes"])*100)
print("% of non Backorders where there was a potential issue and oe constraint",len(data[(data.went_on_backorder=="No")&(data.potential_issue=="Yes")&(data.oe_constraint =="Yes")])/len(data[data.went_on_backorder=="No"])*100)





plt.figure(figsize=(15,5))

plt.subplot(1,2,1)
sns.countplot(x=data[data.went_on_backorder=="No"].potential_issue,hue=data[data.went_on_backorder=="No"].rev_stop)
plt.title("Relation between rev stop and potential issues\nFor Non Backordered Products")

plt.subplot(1,2,2)
sns.countplot(x=data[data.went_on_backorder=="Yes"].potential_issue,hue=data[data.went_on_backorder=="Yes"].rev_stop)
plt.title("Relation between rev stop and potential issues\nFor Backordered Products")

plt.show()

print("% of Backorders where there was a potential issue and rev stop",len(data[(data.went_on_backorder=="Yes")&(data.potential_issue=="Yes")&(data.rev_stop =="Yes")])/len(data[data.went_on_backorder=="Yes"])*100)
print("% of non Backorders where there was a potential issue and rev stop",len(data[(data.went_on_backorder=="No")&(data.potential_issue=="Yes")&(data.rev_stop =="Yes")])/len(data[data.went_on_backorder=="No"])*100)





plt.figure(figsize=(15,5))

plt.subplot(1,2,1)
sns.countplot(x=data[data.went_on_backorder=="No"].potential_issue,hue=data[data.went_on_backorder=="No"].stop_auto_buy)
plt.title("Relation between stop auto buy and potential issues\nFor Non Backordered Products")

plt.subplot(1,2,2)
sns.countplot(x=data[data.went_on_backorder=="Yes"].potential_issue,hue=data[data.went_on_backorder=="Yes"].stop_auto_buy)
plt.title("Relation between stop auto buy and potential issues\nFor Backordered Products")

plt.show()

print("% of Backorders where there was a potential issue and auto buy stop",len(data[(data.went_on_backorder=="Yes")&(data.potential_issue=="Yes")&(data.stop_auto_buy =="Yes")])/len(data[data.went_on_backorder=="Yes"])*100)
print("% of non Backorders where there was a potential issue auto buy stop",len(data[(data.went_on_backorder=="No")&(data.potential_issue=="Yes")&(data.stop_auto_buy =="Yes")])/len(data[data.went_on_backorder=="No"])*100)


#### Observations
#
# - Based on the correlation heatmap, we observed that the dataset has a huge amount of multicollinearity with more than 30 pairs of numerical features having a correlation of 0.80 or higher.
# - We can see features like sales, forecasts and performance which are given for multiple months i.e 1, 3, 6, 9 or 12 months. However, all of them mainly have they same type and magnitude of correlationship with other features. Also, these features are contributing the most to the multi-collinearity.
# - On analysing the variance inflation factors, we can see that the vif for these features is beyond accepted levels of 10, which is causing high multi-collinearity.
# - Hence for sales, forecasts and performance, we will keep 1 features for each and get rid of the other similar features.
#
#
# - In terms of analysis, when understanding the relation of minimum recommended stock of product to hold with stock in transit with respect to backorders and non-backorders, we observed that in for backorders, there we many cases where the minimum stock recommended to hold was way more than the stock that was in circulation.
# - Also, around 12% of backordered products observed more no. of pieces past due than the total national inventory for the product. The figure was less than 1% for non backordered items.
# - Also, we noticed that for 12 % of backordered items, the national level of inventory has remained same irrespective of the amount of stock in transition. For non backordered items, the number was only 3%.
# - Also, we noticed that for 33% of backordered items, the national level inventory for the product was lower than the minimum stock recommended to hold. The figure was only 3% In case of non backordered items.
#
#
# - Within categorical features, we can observed that there were a few backordered items which had a potential issue flag and ppap risk, deck risk or risk of stopping auto buy attached with them. For non backordered items, such risks was significantly quite low.
#
#
# **At the end of the EDA, we can claim that based on %, graphical and statistical analysis, the features we have are useful to clearly distinguish between the two classes. However, the imbalance in the target classes needs to be corrected.**

data.drop(["sales_1_month","sales_3_month","sales_9_month","forecast_3_month","forecast_9_month","perf_12_month_avg"],axis=1,inplace=True)
logger.info("Dropping high correlated independent numerical features causing high multi-collinearity")


# Checking correlation heatmap and vif after reducing multicollinearity
plt.figure(figsize=(12,10))
sns.heatmap(round(data.corr(),2),annot=True)
plt.title("Correlation Heatmap",size=15)
plt.show()


num_cols = data.select_dtypes(include=np.number).drop("lead_time",axis=1)
vif = [variance_inflation_factor(num_cols.values,i) for i in range(len(num_cols.columns))]
VIF = pd.DataFrame({"Feature":num_cols.columns,"vif":vif})
VIF.sort_values("vif",ascending=False,inplace=True)
VIF


logger.info("Exploratory Data Analysis completed")


### Seperating Predictors and Predicted variables.

X = data.drop("went_on_backorder",axis=1)
y = data.went_on_backorder

X.to_csv("Xsample.csv")
y.to_csv("Ysample.csv")
