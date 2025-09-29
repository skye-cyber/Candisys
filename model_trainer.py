from __future__ import print_function

import argparse
import logging
import os
import pickle
import warnings
import dask.dataframe as dd
import lightgbm as lgb

# import matplotlib.pyplot as plt
# import numpy as np
import pandas as pd

# import plotly.express as px
# import seaborn as sns
from colorama import Fore, Style, init

# from sklearn import metrics, tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import cross_val_score, train_test_split

# from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from config import getLogger, open_dataset, MODEL_DIR

logger = getLogger()

warnings.filterwarnings("ignore")


logger.info(f"Dataset: \033[1;94m{open_dataset}\033[0m")

df = pd.read_csv(open_dataset)

# Dask DataFrame
df = dd.read_csv(open_dataset)
# Converting Dask DataFrame to pandas for further operations
df = df.compute()

"""INSPECT
# print(df.head())  # csv head
# print(df.shape)  # Number of columns
# print(df.columns)  # Column names
# print(df['Suitable'].unique())  # get a column
# print(df.dtypes)  # Column datatype
# print(df['Suitable'].value_counts())  # Number of crop occurences


# Get the min and max for each numerical column
# min_values = df.min()
# max_values = df.max()

# print("Minimum values:\n", min_values)
# print("\nMaximum values:\n", max_values)

pd.set_option('display.max_columns', 28)  # Use max of 28 columns

# Group by 'ap_Age' and calculate key statistics for numerical columns
key_stats = df.groupby("ap_Age").agg(
    {
        "ap_Education": ["mean", "min", "max", "std"],
        "ap_YearsOfExperience": ["mean", "min", "max", "std"],
        "ap_TechnicalScore": ["mean", "min", "max", "std"],
        "ap_InterviewScore": ["mean", "min", "max", "std"],
    }
)

# Optional: flatten the MultiIndex columns
key_stats.columns = ['_'.join(col).strip() for col in key_stats.columns.values]

print(key_stats)"""

"""/INSPECT"""

"""Separating Features and Target Label"""
# every column except the last
features = df.iloc[:, :-1]
target = df["Suitable"]

# Splitting into train and test data

X_train, X_test, y_train, y_test = train_test_split(
    features, target, test_size=0.2, random_state=2
)

# print(X_train)
"""/Separation"""

# Initializing empty lists to append all model's name and corresponding name
# This is required later on when defining the function that helps in the DRY principle

accuracy_list = []

model_list = []


# Function To Avoid Dry
def train_and_save_model(
    model, model_name, X_train, y_train, X_test, y_test, features, target, save_path
):
    """
    Function to train a model, print performance metrics with decorative separators,
    perform cross-validation, and save the trained model to a file. It also stores
    accuracy scores and model names.

    Parameters:
    - model: The model to train.
    - model_name: A string name of the model.
    - X_train: Training features.
    - y_train: Training data.
    - X_test: Testing features.
    - y_test: Testing data.
    - features: Features for cross-validation.
    - target: Target for cross-validation.
    - save_path: Path where the model should be saved.
    """
    separator = "-" * 50  # Decorative separator

    save_path = os.path.join(MODEL_DIR, save_path)

    print(separator)

    logger.info(f"\033[0mTraining \033[32m{model_name}\033[0m...\n{separator}")

    # Fit the model
    model.fit(X_train, y_train)

    # Predict on the test set
    predicted_values = model.predict(X_test)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, predicted_values)

    accuracy_list.append(accuracy)

    model_list.append(model_name)

    print(f"{model_name}'s Accuracy is: {accuracy*100:.2f}%\n{separator}")

    # Print classification report
    print(f"Classification Report for {model_name}:\n{separator}")

    print(classification_report(y_test, predicted_values))

    # Perform cross-validation
    score = cross_val_score(model, features, target, cv=5)

    print(
        f"{separator}\nCross-validation scores for {model_name}: {score}\n{separator}"
    )

    # Save the model
    with open(save_path, "wb") as model_pkl:
        pickle.dump(model, model_pkl)

    logger.info(
        f"\033[1;34m{model_name}\033[0m model saved to \033[1;34m{save_path}\033[0m\n{separator}"
    )


def _DecisionTree():
    """DECISION TREE"""
    # criterion = ('gini', 'entropy')
    DecisionTree = DecisionTreeClassifier(
        criterion="gini", max_leaf_nodes=90, random_state=2, max_depth=11
    )  # -> 90.30%

    train_and_save_model(
        DecisionTree,
        "Decision Tree",
        X_train,
        y_train,
        X_test,
        y_test,
        features,
        target,
        "DecisionTree.pkl",
    )
    """DecisionTree.fit(X_train, y_train)

    predicted_values = DecisionTree.predict(X_test)

    x = metrics.accuracy_score(y_test, predicted_values)

    accuracy_list.append(x)

    model_list.append('Decision Tree')

    # print("Decision Tree's Accuracy is: ", x*100)

    # print(classification_report(y_test, predicted_values))


    # Cross validation score (Decision Tree)

    score = cross_val_score(DecisionTree, features, target, cv=5)
    # print(score)


    # Saving trained Decision Tree model

    logger.info("Saving: DecisionTree.pkl")
    with open('DecisionTree.pkl', 'wb') as DT_model_pkl:
        pickle.dump(DecisionTree, DT_model_pkl)"""


def _LightGBM():
    """
    LightGBM (Light Gradient Boosting Machine) is a highly efficient, distributed,
    and fast implementation of the Gradient Boosting framework. It's designed to
    be highly scalable and to handle large datasets with high-dimensional features,
    making it especially popular in machine learning competitions like those hosted by Kaggle.

    LightGBM is based on the Gradient Boosting algorithm, which builds models
    sequentially by combining weak learners (typically decision trees) to minimize
    a loss function.
    Each new model focuses on the errors of the previous models, gradually improving
    prediction accuracy.
    PARAMETERS
    num_leaves:

    The number of leaves in one tree. Increasing the number of leaves can improve
    accuracy but may lead to overfitting.

    learning_rate: The step size for updating weights. Lower values make the model more robust but increase the training time.

    max_depth: Maximum depth of trees. Shallower trees prevent overfitting but might reduce accuracy.

    min_data_in_leaf: The minimum number of data points required to be in a leaf. This helps control overfitting by preventing leaves from becoming too specific to individual samples.

    boosting_type: The boosting strategy, which could be gbdt (Gradient Boosting Decision Tree), dart (Dropouts meet Multiple Additive Regression Trees), or goss (Gradient-based One-Side Sampling).

    """

    # n_estimators=100, learning_rate=0.1, num_leaves=31, random_state=42
    LightGBM = lgb.LGBMClassifier(
        verbose=-1, num_leaves=70, n_estimators=11
    )  # -> 88.76%
    train_and_save_model(
        LightGBM,
        "Light Gradient Boosting Machine",
        X_train,
        y_train,
        X_test,
        y_test,
        features,
        target,
        "LightGBM.pkl",
    )


def _SVM():
    """
    Support Vector Machine (SVM)
        ->A Support Vector Machine (SVM) is a powerful supervised machine learning
        algorithm used for both classification and regression tasks, although it is
        most commonly used for classification.
        ->SVM works by finding the hyperplane that best separates the data into two
        classes. In two dimensions, this hyperplane is a line, but in higher dimensions,
        it can be a plane or a hyperplane
    """

    SVM = SVC(gamma="auto", kernel="linear")  # 89.48%

    train_and_save_model(
        SVM,
        "Support Vector Machine",
        X_train,
        y_train,
        X_test,
        y_test,
        features,
        target,
        "SVM.pkl",
    )


def _Logistic_Regression():
    """
    Logistic Regression
        ->Logistic Regression is a supervised machine learning algorithm used
        primarily for binary classification tasks, though it can be extended for
        multiclass classification as well. Despite its name, logistic regression is
        a classification algorithm, not a regression algorithm. It predicts the
        probability that a given data point belongs to a particular class.
    """
    LogReg = LogisticRegression(random_state=2, C=0.1, max_iter=10_000)  # 86.47% ->
    train_and_save_model(
        LogReg,
        "Logistic Regression",
        X_train,
        y_train,
        X_test,
        y_test,
        features,
        target,
        "LogReg.pkl",
    )


def _RFC():
    """
    Random Forest Classifier
        ->Random forest combines the predictions of multiple decision trees to
        produce a final prediction. This helps reduce overfitting, improves
        accuracy, and increases stability.

    PARAMETERS
    n_estimators: Number of decision trees to build (more trees can improve
                    accuracy but increase computation time).
    max_depth: Maximum depth of the trees (restricts the depth to prevent overfitting).
    criterion: Function to measure the quality of the split (gini or entropy).
    max_features: The number of features to consider when looking for the best split.

    """
    RFC = RandomForestClassifier(
        n_estimators=55, max_leaf_nodes=50_000_000, random_state=42
    )  # 90.62%

    train_and_save_model(
        RFC,
        "Random Forest Classifier",
        X_train,
        y_train,
        X_test,
        y_test,
        features,
        target,
        "RFClassifier.pkl",
    )


def _XGBoost():
    # Initialize LabelEncoder
    crop_encoder = LabelEncoder()

    # Encode the target variable
    df_encoded = df.copy()

    encoder = LabelEncoder()
    df_encoded = df.copy()

    # Encode labels from str-> models only understand numbers
    df_encoded["Suitable"] = encoder.fit_transform(df_encoded["Suitable"])
    df_encoded["em_PreviouEmployment"] = encoder.fit_transform(
        df_encoded["em_PreviouEmployment"]
    )
    df_encoded["ap_PreviouEmployment"] = encoder.fit_transform(
        df_encoded["ap_PreviouEmployment"]
    )
    # Separate encoded target
    target_encoded = df_encoded["Suitable"]

    features_encoded = df_encoded.iloc[:, :-1]

    # Split the dataset
    X_train, X_test, y_train_encoded, y_test_encoded = train_test_split(
        features_encoded, target_encoded, test_size=0.2, random_state=2
    )

    Xboost = XGBClassifier(eta=0.1, n_estimators=40, max_depth=6)  # 90.72% -> 89.88

    train_and_save_model(
        Xboost,
        "XGBoost Classifier",
        X_train,
        y_train_encoded,
        X_test,
        y_test_encoded,
        features_encoded,
        target_encoded,
        "XGBoost.pkl",
    )


def main():
    parser = argparse.ArgumentParser(description="Train Crop_recommendation models")
    parser.add_argument(
        "--model",
        "-m",
        choices=["xgboost", "RFC", "LR", "SVM", "lightgbm", "decision_tree"],
        type=str,
        help="Choose model to train default=all",
    )
    # parser.add_argument('--verbose', '-v', action="store_true", help="Show alot of information")

    args = parser.parse_args()
    # verbose = args.verbose
    model = args.model

    try:
        if model:
            if model == "xgboost":
                _XGBoost()

            elif model == "RFC":
                _RFC()

            elif model == "LR":
                _Logistic_Regression()

            elif model == "SVM":
                _SVM()
            elif model == "lightgbm":
                _LightGBM()
            elif model == "decision_tree":
                _DecisionTree()

        else:
            _DecisionTree()
            _LightGBM()
            _XGBoost()
            _RFC()
            _Logistic_Regression()
            _SVM()
    except KeyboardInterrupt:
        logger.info("\033[0;1;30mQuit!\033[0m")
        exit(0)

    except Exception as e:
        raise
        logger.error(e)


if __name__ == "__main__":
    main()
