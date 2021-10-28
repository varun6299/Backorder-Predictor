
# Importing required input and output data
xtrain = pd.read_csv("xtrain.csv")
xtest = pd.read_csv("xtest.csv")
ytrain = pd.read_csv("ytrain.csv")
ytest = pd.read_csv("ytest.csv")
logger.info("Train and Test Samples Imported")


# dropping unwanted columns
xtrain.drop(["Unnamed: 0"],axis=1,inplace=True)
xtest.drop(["Unnamed: 0"],axis=1,inplace=True)
ytrain.drop(["Unnamed: 0"],axis=1,inplace=True)
ytest.drop(["Unnamed: 0"],axis=1,inplace=True)


from sklearn.preprocessing import StandardScaler


## Building ML Models

logger.info("Building Baseline")

## Importing Required libraries, packages, functions for model building
import pickle
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier,GradientBoostingClassifier

from sklearn.metrics import f1_score,accuracy_score,recall_score,classification_report,confusion_matrix,roc_auc_score,roc_curve

from sklearn.model_selection import cross_val_score, GridSearchCV

import datetime

logger.info("Imported required functions for building models, evaluation and further model tuning")


# In[214]:


get_ipython().system('pip install xgboost')


from xgboost import XGBClassifier

nums = ['national_inv', 'lead_time', 'in_transit_qty', 'forecast_6_month',
       'sales_6_month', 'min_bank', 'pieces_past_due', 'perf_6_month_avg',
       'local_bo_qty']


# Creating dataframe to store results of baseline models
baseline_results = pd.DataFrame({"Model":None,"Train F1 score":None,"Test F1":None,"ROC-AUC":None,"CV F1 average":None,"CV F1 std average":None,"Model Latency for 1 unit":None},index=range(1))


# Creating a function to automate task of model building & evaluation
def build_baseline_model(estimator,x_training,y_training,x_testing,y_testing,scaling=False,num_cols = None):

    """
    Creates a ML Model for given estimator and train sample and tests on test sample. Performs cross validation too. Uses f1 score for evaluation.

    Note - It is assumed that missing values treatment, outlier treatment, transformation and encoding have already been performed.

    Arguments -

    1. num_cols : A list of numerical features in the dataset. If not passed, function will try to find numerical features

    """
    if str(f"{estimator}").replace("()","") == "XGBClassifier":
        model_str = "XGBClassifier"
    else:
        model_str = str(f"{estimator}").replace("()","")


    if num_cols == None:
        nums_cols = list(x_training.select_dtypes(include=np.object).columns)


    if scaling == True:
        from sklearn.preprocessing import StandardScaler
        SC = StandardScaler().fit(x_training[list(num_cols)])
        x_training[list(nums)] = SC.transform(x_training[list(num_cols)])
        x_testing[list(nums)] = SC.transform(x_testing[list(num_cols)])

    ## Training the model and calculating f1 scores on train and test along with cross validation scores

    try:

        model = estimator.fit(x_training,y_training)
        train_pred = model.predict(x_training)
        logger.info(f"Training {model_str} Model complete")
        test_pred = model.predict(x_testing)
        cv = cross_val_score(estimator,x_training,y_training,scoring="f1",cv=3)
        train_f1 = f1_score(y_training,train_pred)
        test_f1 = f1_score(y_testing,test_pred)
        cv_mean = cv.mean()
        cv_std = cv.std()

        ## Model Latency for 1 unit in seconds
        start = datetime.datetime.now()
        model.predict(x_training.iloc[0].values.reshape(1,-1))
        end = datetime.datetime.now()
        latency = (end.microsecond - start.microsecond) / 1000000


        ## Calculating ROC AUC Score
        test_pred_prob = model.predict_proba(x_testing)[:,1]
        roc = roc_auc_score(y_testing,test_pred_prob)


        ## Appending results to model performance table

        try:
            global baseline_results

            baseline_results = baseline_results.append({"Model":model_str,"Train F1 score":train_f1,"Test F1":test_f1,"ROC-AUC":roc,"CV F1 average":cv_mean,"CV F1 std average":cv_std,"Model Latency for 1 unit":latency},ignore_index=True)
        except NameError:
            print("Baseline results table not defined")

        ## Displaying results
        print(f"Model Performance - {model_str}")

        print("Train Result:",classification_report(y_training,train_pred),sep="\n")
        print()
        print(f"Average f1 Score: {cv_mean}",f"Average Std in F1 score: {cv_std}",sep="\n")
        print()
        print("Test Result:",classification_report(y_testing,test_pred),sep="\n")

        fpr,tpr,thres = roc_curve(y_testing,test_pred_prob)
        plt.plot(fpr,tpr)
        plt.fill_between(fpr,tpr,alpha=0.3)
        plt.title(f"AUC Curve for {model_str} \nROC-AUC Score: {roc}",size=15)
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.show()
        print("","Model training and testing completed","",sep="\n")

    except NameError:
        print("Either the estimator or one or more of the samples are not defined or loaded")
        logging.error(f"Either the {model_str} estimator or one or more of the samples are not defined or loaded")
    else:
        print("Function End Reached")

logger.info("Function Built to create and run baseline models")

### Logistic Regression
build_baseline_model(LogisticRegression(),xtrain,ytrain,xtest,ytest,num_cols=nums)


### K Neighbors Classifier
build_baseline_model(KNeighborsClassifier(),xtrain,ytrain,xtest,ytest,num_cols=nums,scaling=True)


### Decision Tree
build_baseline_model(DecisionTreeClassifier(random_state=0),xtrain,ytrain,xtest,ytest,num_cols=nums)


### Support Vector Machine
build_baseline_model(SVC(probability=True),xtrain,ytrain,xtest,ytest,num_cols=nums)


### Random Forest Classifier
build_baseline_model(RandomForestClassifier(random_state=0),xtrain,ytrain,xtest,ytest,num_cols=nums)


### Ada Boost Classifier
build_baseline_model(AdaBoostClassifier(random_state=0),xtrain,ytrain,xtest,ytest,num_cols=nums)


### Gradient Boosting Classifier
build_baseline_model(GradientBoostingClassifier(random_state=0),xtrain,ytrain,xtest,ytest,num_cols=nums)


### XGB Classifier
build_baseline_model(XGBClassifier(),xtrain,ytrain,xtest,ytest,num_cols=nums)


baseline_results.dropna(inplace=True)
baseline_results


# Inferences -
#
# - Based on the analysis of f1 scores, cross validation and model latency for several models, we will decide which model to choose for further tuning.
# - In terms of F1 scores across train and test, XGB Classifier, Random Forest Classifier and K Nearest Neighbors are providing scores around 0.94-0.96, with Random Forest Providing the best result on test i.e 0.954
# - In terms of Area Under Curve scores, XGB Classifier, Gradient Boosting and Random Forest Classifier are providing scores aroud 0.99, with XGB Classifier and Random Forest with identical scores of 0.994.
# - In terms of cross validation, Random Forest Provides the best result across the 3 training samples i.e 0.95. However,Gradient Boosting classifier provides the lowest deviation in f1 scores i.e 0.001807. Amongst other ensemble techniques,Random Forest provides the lowest deviation i.e 0.0023.
# - In terms of latency i.e amount of time model takes to make a prediction for a given input, single estimators like Decision Tree, Logistic regression and Support Vector take the lowest time.
# - Random Forest takes the highest amount of time. However, using feature selection and parameter tuning, we can reduce latency time.
# - Based on comparisons, we can see that Random Forest is the best performer and hence, we will choose Random Forest for further tuning
