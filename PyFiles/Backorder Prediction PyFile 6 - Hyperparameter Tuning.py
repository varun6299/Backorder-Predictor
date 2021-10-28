## Grid Search for Hyperparameter tuning

# Defining parameters
params = {"n_estimators":[20,40,60,80],"max_depth":range(3,11),"criterion":["gini","entropy"],"min_samples_leaf":[20,30,40,50],"min_samples_split":[60,50,40]}


# Initializing grid search
logger.info("Hyperparameter Tuning started using Grid Search")
grid = GridSearchCV(RandomForestClassifier(random_state=0),params,cv=3,scoring="recall").fit(xtrain[top10],ytrain)
grid.best_params_

# Saving the best parameters
best_params = {"criterion":"entropy",'max_depth': 10, 'n_estimators': 60,"min_samples_leaf":20,"min_samples_split":50}
best_params


# Saving the best parameters in a byte stream format.
pickle.dump(best_params,open("best_params.pkl","wb"))


# Running the final model with best parameters and evaluating performance.
tuned_rf_importances = RandomForestClassifier(max_depth=10,n_estimators = 60,random_state=0,criterion="entropy",min_samples_leaf=20,min_samples_split=50).fit(xtrain[top10],ytrain)
logger.info("Training tuned rf classifier with importances features")
tuned_rf_importances_train_pred = tuned_rf_importances.predict(xtrain[top10])
tuned_rf_importances_test_pred = tuned_rf_importances.predict(xtest[top10])
tuned_rf_importances_cv = cross_val_score(RandomForestClassifier(max_depth=10,n_estimators = 60,random_state=0,criterion="entropy",min_samples_leaf=20,min_samples_split=50),xtrain[top10],ytrain,scoring="recall",cv=3)
tuned_rf_importances_train_f1 = f1_score(ytrain,tuned_rf_importances_train_pred)
tuned_rf_importances_test_f1 = f1_score(ytest,tuned_rf_importances_test_pred)
tuned_rf_importances_recall = recall_score(ytest,tuned_rf_importances_test_pred)
tuned_rf_importances_cv_mean = tuned_rf_importances_cv.mean()
tuned_rf_importances_cv_std = tuned_rf_importances_cv.std()

## Model Latency for 1 unit
start = datetime.datetime.now()
tuned_rf_importances.predict(xtrain[top10].iloc[0].values.reshape(1,-1))
end = datetime.datetime.now()
latency = (end.microsecond - start.microsecond) / 1000000


print("Train Result:",f"Latency: {latency}",classification_report(ytrain,tuned_rf_importances_train_pred),sep="\n")
print()
print(f"Average Recall Score: {tuned_rf_importances_recall}",f"Average Std in recall score: {tuned_rf_importances_cv_std}",sep="\n")
print()
print("Test Result:",classification_report(ytest,tuned_rf_importances_test_pred),sep="\n")

tuned_rf_importances_test_pred_prob = tuned_rf_importances.predict_proba(xtest[top10])[:,1]
tuned_rf_importances_roc = roc_auc_score(ytest,tuned_rf_importances_test_pred_prob)
fpr,tpr,thres = roc_curve(ytest,tuned_rf_importances_test_pred_prob)
plt.plot(fpr,tpr)
plt.fill_between(fpr,tpr,alpha=0.3)
plt.title(f"AUC Curve \nROC-AUC Score: {tuned_rf_importances_roc}",size=15)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.show()

#### Observations -
#
# - On tuning the Random Forest with top 10 features, the performance of the model dropped slightly indicated by the lower ROC-AUC Score, lower recall, lower f1 train and test scores.
# - Also, from the roc-auc curve, we can see that the roc-auc score has dropped to 0.9922 from 0.994.
# - While the recall on test has remained same, we can see higher deviation in recall scores during cross validation have increased from 0.0028 to 0.0055 and average recall has dropped from approx 0.93 to 0.90.
# - Though latency has reduced by from 0.015 to 0.011 however, performances have reduced.
# - We cannot afford too much compromise on performance for few microseconds as increased false negatives might cause a business to lose potential customers and sales and hence we will continue using the baseline random forest model with top 10 features.


model = RandomForestClassifier(random_state=0).fit(xtrain[top10],ytrain)

# Saving the final model in a byte stream format using pickle
pickle.dump(model,open("rf_importances.pkl","wb"))

# Saving list of feature selected for final model in a byte stream using pickle.
pickle.dump(top10,open("Final_Features.pkl","wb"))
