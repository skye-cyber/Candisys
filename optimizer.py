import pandas as pd
from sklearn.utils import resample
from imblearn.over_sampling import SMOTE
from config import open_dataset
from sklearn.preprocessing import LabelEncoder


class Optimizer:
    def __init__(self, df: pd.DataFrame, target_col: str):
        """
        Initialize the Optimizer with dataset and target column.

        Args:
            df (pd.DataFrame): Input dataset.
            target_col (str): Column name of target variable (e.g. 'Suitable').
        """
        self.df = df
        self.target_col = target_col

    def oversample(self, random_state: int = 42):
        """Oversample the minority class."""
        majority = self.df[
            self.df[self.target_col] == self.df[self.target_col].mode()[0]
        ]
        minority = self.df[
            self.df[self.target_col] != self.df[self.target_col].mode()[0]
        ]

        minority_upsampled = resample(
            minority,
            replace=True,
            n_samples=len(majority),
            random_state=random_state,
        )

        return (
            pd.concat([majority, minority_upsampled])
            .sample(frac=1, random_state=random_state)
            .reset_index(drop=True)
        )

    def undersample(self, random_state: int = 42):
        """Undersample the majority class."""
        majority = self.df[
            self.df[self.target_col] == self.df[self.target_col].mode()[0]
        ]
        minority = self.df[
            self.df[self.target_col] != self.df[self.target_col].mode()[0]
        ]

        majority_downsampled = resample(
            majority,
            replace=False,
            n_samples=len(minority),
            random_state=random_state,
        )

        return (
            pd.concat([majority_downsampled, minority])
            .sample(frac=1, random_state=random_state)
            .reset_index(drop=True)
        )

    def smote(self, random_state: int = 42):
        """Apply SMOTE to balance dataset."""
        X = self.df.drop(columns=[self.target_col])
        y = self.df[self.target_col]

        # Encode categorical features
        X_encoded = encode_features(X)

        smote = SMOTE(random_state=random_state)
        X_res, y_res = smote.fit_resample(X_encoded, y)

        return pd.concat([X_res, y_res], axis=1)


def encode_features(df, exclude=[]):
    df_encoded = df.copy()
    le = LabelEncoder()
    for col in df_encoded.columns:
        if col not in exclude and df_encoded[col].dtype == object:
            df_encoded[col] = le.fit_transform(df_encoded[col])
    return df_encoded


if __name__ == "__main__":
    df = pd.read_csv(open_dataset).drop_duplicates()
    cleaned_df = (
        df.iloc[:, 1:] if df.columns[0].lower() in ["Unnamed: 0", "index"] else df
    )
    optimizer = Optimizer(cleaned_df, target_col="Suitable")

    balanced_over = optimizer.oversample()
    balanced_under = optimizer.undersample()
    balanced_smote = optimizer.smote()

    print("Oversampled:", balanced_over["Suitable"].value_counts())
    print("Undersampled:", balanced_under["Suitable"].value_counts())
    print("SMOTE:", balanced_smote["Suitable"].value_counts())
