from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler

from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import LocalOutlierFactor, KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer
from skopt.sampler import Grid
from imblearn.pipeline import Pipeline
from imblearn import FunctionSampler
import pandas as pd
import numpy as np
from sklearn.metrics import recall_score, precision_score,accuracy_score

df = pd.read_csv('breast-cancer.csv')

# drop the id column
df.drop(columns='id', inplace=True)

# find the outliers
y_pred = LocalOutlierFactor().fit_predict(df.drop(["diagnosis"], axis=1))

outlier_count = abs(sum(y_pred[y_pred < 1]))
print(f'The vanilla Local Outlier Factor identified {outlier_count} outliers ({round(outlier_count/len(df), 2)}%)' )
# encoding the target variable  
df['diagnosis'] = df['diagnosis'].map({'B' : 0, 'M' : 1})

# convert it into a numeric variable
df['diagnosis'] = pd.to_numeric(df['diagnosis'])

# create the correlation matrix
corr_mat = df.corr(method='spearman')
high_corr_feat_list = list(corr_mat[(abs(corr_mat['diagnosis']) >= 0.5) & (corr_mat.columns != 'diagnosis')].index)

"""Start Features"""

# extract the target variable
y = df['diagnosis']
# split the columns into features and targets
X_full = df[df.columns.drop(['diagnosis'])]
# split the dataset into 3 sets
X_mean = df.iloc[:, 1:11]
X_se = df.iloc[:, 11:21]
X_worst = df.iloc[:, 21:]
# Reduced features
X_high_corr = X_full[high_corr_feat_list]
X_hand_picked = X_full[['radius_mean','concave points_mean','radius_se','texture_mean']]
X_radius = X_full[['radius_mean','radius_se']]

"""End Features"""

def nested_cv(X: pd.DataFrame, y: pd.Series, cv_outer: StratifiedKFold, opt_search: BayesSearchCV, validation_result: bool) -> None:
    """
    Run the nested cross validation.
    """
    outer_results, inner_results, outer_precisions, outer_accuracies = outer_loop(X=X, y=y, cv_outer=cv_outer, opt_search=opt_search, validation_result=validation_result)

    # print the CV overall results
    print(f'Recall | Validation Mean: {round(np.mean(inner_results), 3)}, Validation Std: {round(np.std(inner_results), 3)}')
    print(f'Recall | Test Mean: {round(np.mean(outer_results), 3)}, Test Std: {round(np.std(outer_results), 3)}')
    print(f'Precision | Test Mean: {round(np.mean(outer_precisions), 3)}, Test Std: {round(np.std(outer_precisions), 3)}')
    print(f'Accuracy | Test Mean: {round(np.mean(outer_accuracies), 3)}, Test Std: {round(np.std(outer_accuracies), 3)}')

def outer_loop(X: pd.DataFrame, y: pd.Series, cv_outer: StratifiedKFold, opt_search: BayesSearchCV, validation_result: bool) -> list:
    """
    Perform the outer loop split and per each fold, its inner loop.
    """
    outer_results, inner_results , outer_precisions, outer_accuracy= [], [], [],[]
    
    for i, (train_index, test_index) in enumerate(cv_outer.split(X, y), start=1):
        X_train, X_test = X.loc[train_index], X.loc[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        # start the Bayes search
        _ = opt_search.fit(X_train, y_train)

        # save the best model
        best_model = opt_search.best_estimator_
        # predict on the test set
        y_pred = best_model.predict(X_test)

        # calculate the recall on test set
        recall = recall_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        accuracy = accuracy_score(y_test,y_pred)
        # append the recall results
        outer_results.append(recall)
        inner_results.append(opt_search.best_score_)
        outer_precisions.append(precision)
        outer_accuracy.append(accuracy)
        print_validation_results(i=i, opt_search=opt_search, recall=recall, validation_result=validation_result)
    
    return outer_results, inner_results, outer_precisions, outer_accuracy


def print_validation_results(i: int, opt_search: BayesSearchCV, recall: float, validation_result: bool) -> None:
    """
    Print the validation results per each fold.
    """
    if validation_result:
        print(f'Fold {i}')
        print(f'Recall | Validation: {round(opt_search.best_score_, 3)}\tTest: {round(recall, 3)}')
        print('\n')
        print(f'Best Hyperparameter Combination:\n{opt_search.best_params_}')
        print('\n')
        

# set the inner and outer CV
cv_outer = StratifiedKFold(n_splits=10, shuffle=True, random_state=1)
cv_inner = StratifiedKFold(n_splits=10, shuffle=True, random_state=1)
# set the dict with params to
# be passed onto the optimizer
optimizer_dict = {
        'n_initial_points' : 10,
        'initial_point_generator' : Grid(border="include")
        }

# create the pipeline
pipeline = Pipeline([
    ('clf', None)
])

# set the parameter search space for classifier 1
gnb_search = {
    'clf': Categorical([GaussianNB()]),
    'clf__var_smoothing': Real(1e-9, 2)
}

# set the parameter search space for classifier 2
svc_search = {
    'clf': Categorical([SVC()]),
    'clf__C': Real(1e-6, 1e+6, prior='log-uniform'),
    'clf__gamma': Real(1e-6, 1e+1, prior='log-uniform'),
    'clf__degree': Integer(1, 3),
    'clf__kernel': Categorical(['linear', 'poly', 'rbf'])
}

# set the parameter search space for classifier 3
log_search = {
    'clf': Categorical([LogisticRegression()]),
    'clf__C': Real(1e-5, 10),
    'clf__penalty': Categorical(['l2']),
    'clf__class_weight': Categorical([None, 'balanced']),
    'clf__solver': Categorical(['lbfgs', 'liblinear']),
    'clf__max_iter': [1000]
}

# set the parameter search space for classifier 4
rf_search = {
    'clf': Categorical([RandomForestClassifier()]),
    'clf__n_estimators': Integer(10, 200),
    'clf__criterion': Categorical(['gini', 'entropy']),
    'clf__min_samples_split': Integer(2, 200),
    'clf__min_samples_leaf': Integer(1, 200),
    'clf__min_impurity_decrease': Real(0, 1),
    'clf__max_features': Integer(1, 15)
}

# set the parameter search space for classifier 1
knn_search = {
    'clf': Categorical([KNeighborsClassifier()]),
    'clf__n_neighbors': Integer(2, 20, prior='log-uniform'),
    'clf__weights': Categorical(['uniform', 'distance']),
    'clf__leaf_size': Integer(30, 100),
    'clf__p': Integer(1, 2),
    'clf__algorithm': Categorical(['ball_tree', 'kd_tree', 'brute'])
}


# create a list of models
model_list = [
    gnb_search,
    svc_search,
    log_search,
    rf_search,
    knn_search
]

# set the number of searches to perform
search_num = 10

# create the search space by joining
# all of the classifiers that need to be optimized
search_space_list = [
    (log_search, search_num), 
    (svc_search, search_num), 
    (rf_search, search_num),
    (gnb_search, search_num),
    (knn_search, search_num)
]
## 2nd model
def lof(X, y):
    """Find the outliers above the 1st percentile and remove them from both X and y."""
    model = LocalOutlierFactor()
    model.fit(X)
    # extract 
    lof_score = model.negative_outlier_factor_
    # find the 1st percentile
    percentile = np.quantile(lof_score, 0.01)

    return X[lof_score > percentile, :], y[lof_score > percentile]
# create the pipeline
pipeline = Pipeline([
    ('outlier_detector', FunctionSampler(func=lof)),
    ('scaler', MinMaxScaler()), 
    ('clf', None)
])
# define the Bayesian search
opt_search = BayesSearchCV(
    estimator=pipeline,
    search_spaces=search_space_list,
    optimizer_kwargs=optimizer_dict,
    scoring='recall',
    cv=cv_inner,
    refit=True,
    return_train_score=True,
    random_state=42)
# enumerate splits
print("Full features\n")
nested_cv(X_full, y, cv_outer, opt_search, False)
print("Highly correlated\n")
nested_cv(X_high_corr, y, cv_outer, opt_search, False)
print("Hand picked\n")
nested_cv(X_hand_picked, y, cv_outer, opt_search, False)
print("Radius only\n")
nested_cv(X_radius, y, cv_outer, opt_search, False)
print("Means only\n")
nested_cv(X_mean, y, cv_outer, opt_search, False)
print("Standard error only\n")
nested_cv(X_se, y, cv_outer, opt_search, False)
print("Worst only\n")
nested_cv(X_worst, y, cv_outer, opt_search, False)