## Feature Selection

# Importing mlxtend library for backward feature elimination function - SequentialFeatureSelector
get_ipython().system(' pip install mlxtend')


from sklearn.feature_selection import RFECV
from mlxtend.feature_selection import SequentialFeatureSelector


# Creating dataframe to store results of baseline models
baseline_best_features = pd.DataFrame({"Model":None,"Train F1 score":None,"Test F1":None,"Test Recall":None,"CV Recall average":None,"CV Recall std average":None,"Model Latency for 1 unit in seconds":None},index=range(1))


# Creating a function to automate the task of building model with feature from different feature selection techniques.
def build_feature_selection_model(model_estimator,feature_selector_estimator,x_training,y_training,x_testing,y_testing,scaling=False,num_cols=None):

    """
    Creates a ML Model for given model_estimator and train sample and tests on test sample. Performs cross validation too. Uses f1 score for evaluation.

    Note - It is assumed that missing values treatment, outlier treatment, transformation and encoding have already been performed.

    Also, if unknown technique passed or no technique passed, default technique used is feature importance
    Arguments -

    1. num_cols : A list of numerical features in the dataset. If not passed, function will try to find numerical features

    """
    if str(f"{model_estimator}").replace("()","") == "XGBClassifier":
        model_str = "XGBClassifier"
    else:
        model_str = str(f"{model_estimator}").replace("()","")


    if str(f"{feature_selector_estimator}").split("(")[0] != "TOP10FeatureImportances":
        feature_selector_estimator_str = str(f"{feature_selector_estimator}").split("(")[0]
    else:
        feature_selector_estimator_str = "TOP10FeatureImportances"

    if num_cols == None:
        nums_cols = list(x_training.select_dtypes(include=np.object).columns)

    if scaling == True:
        from sklearn.preprocessing import StandardScaler
        SC = StandardScaler().fit(x_training[list(num_cols)])
        x_training[list(nums)] = SC.transform(x_training[list(num_cols)])
        x_testing[list(nums)] = SC.transform(x_testing[list(num_cols)])


    try:

        # Finding best features

        if feature_selector_estimator_str == "SequentialFeatureSelector":
            SFS = SequentialFeatureSelector(model_estimator,k_features="best",cv=3,scoring="recall",forward=False).fit(x_training,y_training)
            logger.info("Performing Forward Feature Elimination")
            top_features = list(SFS.k_feature_names_)

        elif feature_selector_estimator_str == "RFECV":
            RFE = RFECV(RandomForestClassifier(random_state=0),min_features_to_select=10,cv=3,scoring="recall").fit(x_training,y_training)
            logger.info("Performing Recursive Feature Elimination")
            top_features = list(x_training.columns[RFE.ranking_==1])

        elif feature_selector_estimator_str == "TOP10FeatureImportances":
            print("Since no feature selector technique chosen or unknown selection technique chosen, top 10 features from Feature Importances will be chosen")
            model = model_estimator.fit(x_training,y_training)
            importances = pd.DataFrame({"Feature":x_training.columns,"Importances":model.feature_importances_})
            importances.sort_values(by="Importances",ascending=False,inplace=True)
            logger.info("Calculating Feature Importances")
            top_features = importances.iloc[:10,0].tolist()

        ## Training the model and calculating f1 scores on train and test along with cross validation scores

        model = model_estimator.fit(x_training[top_features],y_training)
        train_pred = model.predict(x_training[top_features])
        logger.info(f"Training {model_str} with {feature_selector_estimator_str} Model complete")
        test_pred = model.predict(x_testing[top_features])
        cv = cross_val_score(model_estimator,x_training[top_features],y_training,scoring="recall",cv=3)
        train_f1 = f1_score(y_training,train_pred)
        test_f1 = f1_score(y_testing,test_pred)
        recall = recall_score(ytest,test_pred)
        cv_mean = cv.mean()
        cv_std = cv.std()

        ## Model Latency for 1 unit in seconds
        start = datetime.datetime.now()
        model.predict(x_training[top_features].iloc[0].values.reshape(1,-1))
        end = datetime.datetime.now()
        latency = (end.microsecond - start.microsecond) / 1000000


        ## Calculating ROC AUC Score
        test_pred_prob = model.predict_proba(x_testing[top_features])[:,1]
        roc = roc_auc_score(y_testing,test_pred_prob)


        ## Appending results to model performance table

        try:
            global baseline_best_features

            baseline_best_features = baseline_best_features.append({"Model":f"{model_str} - {feature_selector_estimator_str}","Train F1 score":train_f1,"Test F1":test_f1,"Test Recall":recall,"CV Recall average":cv_mean,"CV Recall std average":cv_std,"Model Latency for 1 unit in seconds":latency},ignore_index=True)
        except NameError:
            print("Baseline best features results table not defined")

        ## Displaying results
        print(f"Model Performance - {model_str} with features from {feature_selector_estimator_str}")

        print("Train Result:",classification_report(y_training,train_pred),sep="\n")
        print()
        print(f"Average recall Score: {cv_mean}",f"Average Std in recalls score: {cv_std}",sep="\n")
        print()
        print("Test Result:",classification_report(y_testing,test_pred),sep="\n")

        fpr,tpr,thres = roc_curve(y_testing,test_pred_prob)
        plt.plot(fpr,tpr)
        plt.fill_between(fpr,tpr,alpha=0.3)
        plt.title(f"AUC Curve for {model_str} \nROC-AUC Score: {roc}",size=15)
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.show()

        plt.figure(figsize = (6,5))
        sns.heatmap(confusion_matrix(y_testing,test_pred),annot=True,cbar=False,fmt="g")
        plt.xlabel("Predicted Labels")
        plt.ylabel("Actual Labels")
        plt.xticks([0.5,1.5],["Non Backorders","Backorders"])
        plt.yticks([0.5,1.5],["Non Backorders","Backorders"])
        plt.title(f"CONFUSION MATRIX for {model_str} with {feature_selector_estimator_str}")
        plt.show()

        print("","Model training and testing completed","",sep="\n")

    except NameError:
        print("Either the model_estimator or one or more of the samples are not defined or loaded")
        logging.error(f"Either the {model_str} model_estimator, {feature_selector_estimator_str} or one or more of the samples are not defined or loaded")
    except AttributeError:
        print("Model has no feature importance attribute")
        logging.error(f"{model_str} has no feature importance attribute")
    else:
        print("Function End Reached")

    return top_features


### Backward Feature Elimination

sfs = build_feature_selection_model(RandomForestClassifier(random_state=0),SequentialFeatureSelector(RandomForestClassifier(random_state=0),forward=False,k_features="best",cv=3,scoring="recall"),xtrain,ytrain,xtest,ytest)


### Recursive Feature Elimination

rfe = build_feature_selection_model(RandomForestClassifier(random_state=0),RFECV(RandomForestClassifier(random_state=0),min_features_to_select=10,cv=3,scoring="recall"),xtrain,ytrain,xtest,ytest)


### TOP 10 Feature Importances

top10 = build_feature_selection_model(RandomForestClassifier(random_state=0),feature_selector_estimator="TOP10FeatureImportances",x_training=xtrain,y_training=ytrain,x_testing=xtest,y_testing=ytest)



baseline_best_features.dropna(inplace=True)
baseline_best_features


#### Observations -
#
# - We applied the backward feature elimination, recursive feature elimination and feature importance technique using Random Forest.
#
# - On comparing the f1 scores on train and test, recall on test, cross validation and model latency, we can conclude that while test f1 and recall scores are same for all methods, the f1 scores and cross validation recall score for RFE is the best across all models.
#
# - Across all models, despite using lesser or optimum no. of features, the models are taking more time to make a prediction.
#
# - The latency of Random Forest using the top 10 features, compared to the other models using RFE and SFS, is slightly higher.
#
# - However, in terms of interpretation and user interface, using top 10 features is easy as it uses lesser no. of features.
#
# - Also the model using the top 10 features has the lowest deviation in cross validation recalls.
#
# - Also, categorical features from the top 10 can be easily understood as compared to the other models. Other categorical features have no proper explanation, hence making it difficult for users to answer. The categorical features suggested by feature importances and can be interpretated.
#
# - Also, Hence, we will use Random Forest Classifier with top 10 features. We will try to tune the model to reduce latency.
