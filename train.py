import argparse
import os
import pickle
import warnings
import lightgbm as lgb
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from config import getLogger, open_dataset, MODEL_DIR
from optimizer import Optimizer

logger = getLogger()

warnings.filterwarnings("ignore")


logger.info(f"Dataset: \033[1;94m{open_dataset}\033[0m")

df = pd.read_csv(open_dataset)

# Function to separate features and target


def separate_features_target(df):
    optimizer = Optimizer(df, target_col="Suitable")
    blanced_df = optimizer.oversample()

    features = blanced_df.iloc[:, :-1]
    target = blanced_df["Suitable"]
    return features, target


# Function to split the data


def split_data(features, target):
    return train_test_split(features, target, test_size=0.2, random_state=42)


def get_best_param():
    # instantiate the RFC
    rf = RandomForestClassifier(random_state=42)
    features, target = separate_features_target(df)
    X_train, X_test, y_train, y_test = split_data(features, target)
    param_grid = {
        # Number of trees in the forest
        "n_estimators": [10, 20, 50, 100, 200],
        "max_depth": [None, 10, 15, 20, 30],  # minimum depth of trees
        # minimum number of samples required to split a node
        "min_samples_split": [2, 5, 10],
    }
    # set the grid search with 5-fold cross-validation and 100 iterations = n_iter=100,
    # n_jobs=-1 to use all available cpu
    grid_search = GridSearchCV(
        estimator=rf, param_grid=param_grid, cv=5, scoring="accuracy"
    )
    # fit the grid search to the training data
    grid_search.fit(X_train, y_train)

    print("Best Parameters:", grid_search.best_params_)


# Function to handle model training and saving


def train_and_save_model(
    model, model_name, X_train, y_train, X_test, y_test, features, target, save_path
):
    separator = "-" * 50
    save_path = os.path.join(MODEL_DIR, save_path)
    print(separator)

    logger.info(f"\033[0mTraining \033[32m{model_name}\033[0m...\n{separator}")
    model.fit(X_train, y_train)
    predicted_values = model.predict(X_test)
    accuracy = accuracy_score(y_test, predicted_values)

    print(f"{model_name}'s Accuracy is: {accuracy*100:.2f}%\n{separator}")
    print(f"Classification Report for {model_name}:\n{separator}")
    print(classification_report(y_test, predicted_values))

    score = cross_val_score(model, features, target, cv=5)
    print(
        f"{separator}\nCross-validation scores for {model_name}: {score}\n{separator}"
    )

    with open(save_path, "wb") as model_pkl:
        pickle.dump(model, model_pkl)

    logger.info(
        f"\033[1;34m{model_name}\033[0m model saved to \033[1;34m{save_path}\033[0m\n{separator}"
    )


# All models below:


def _DecisionTree(X_train, y_train, X_test, y_test, features, target):
    # criterion: entropy, gini,log_loss
    DecisionTree = DecisionTreeClassifier(
        criterion="gini", max_leaf_nodes=50_900, random_state=2, max_depth=82
    )
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


def _LightGBM(X_train, y_train, X_test, y_test, features, target):
    LightGBM = lgb.LGBMClassifier(verbose=-1, num_leaves=10, n_estimators=11)
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


def _SVM(X_train, y_train, X_test, y_test, features, target):
    # max_iter=-1, kernel=(rbf,auto), gamma=(scale, auto), degree=(3...)
    SVM = SVC(gamma="scale", kernel="linear")
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


def _Logistic_Regression(X_train, y_train, X_test, y_test, features, target):
    # incase of class imbalance
    # model = LogisticRegression(class_weight="balanced")
    LogReg = LogisticRegression(random_state=2, C=0.1, max_iter=9)  # 50.37
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


def _RFC(X_train, y_train, X_test, y_test, features, target):
    RFC = RandomForestClassifier(
        n_estimators=55, max_leaf_nodes=100_000, random_state=42
    )
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


def _XGBoost(df):
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

    features_encoded, target_encoded = separate_features_target(df_encoded)
    X_train, X_test, y_train_encoded, y_test_encoded = split_data(
        features_encoded, target_encoded
    )

    Xboost = XGBClassifier(eta=0.1, n_estimators=40, max_depth=6)
    train_and_save_model(
        Xboost,
        "XGBoost Classifier",
        X_train,
        y_train_encoded,
        X_test,
        y_test_encoded,
        features_encoded,
        target_encoded,
        "XGBoost1.pkl",
    )


def main():
    parser = argparse.ArgumentParser(description="Train Candidate Selection models")
    parser.add_argument(
        "--model",
        "-m",
        choices=["xgboost", "RFC", "LR", "SVM", "lightgbm", "decision_tree"],
        type=str,
        help="Choose model to train default=all",
    )
    args = parser.parse_args()
    model = args.model

    features, target = separate_features_target(df)
    X_train, X_test, y_train, y_test = split_data(features, target)

    try:
        if model:
            if model == "xgboost":
                _XGBoost(df)
            elif model == "RFC":
                _RFC(X_train, y_train, X_test, y_test, features, target)
            elif model == "LR":
                _Logistic_Regression(X_train, y_train, X_test, y_test, features, target)
            elif model == "SVM":
                _SVM(X_train, y_train, X_test, y_test, features, target)
            elif model == "lightgbm":
                _LightGBM(X_train, y_train, X_test, y_test, features, target)
            elif model == "decision_tree":
                _DecisionTree(X_train, y_train, X_test, y_test, features, target)
        else:
            _DecisionTree(X_train, y_train, X_test, y_test, features, target)
            _LightGBM(X_train, y_train, X_test, y_test, features, target)
            _XGBoost(df)
            _RFC(X_train, y_train, X_test, y_test, features, target)
            _Logistic_Regression(X_train, y_train, X_test, y_test, features, target)
            _SVM(X_train, y_train, X_test, y_test, features, target)
    except KeyboardInterrupt:
        logger.info("\033[0;1;30mQuit!\033[0m")
        exit(0)

    except Exception as e:
        raise
        logger.error(e)


if __name__ == "__main__":
    # get_best_param()
    main()
