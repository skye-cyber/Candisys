import pandas as pd
from config import getLogger, open_dataset

logger = getLogger()


def inspect_dataset(path: str):
    logger.info(f"Dataset: \033[1;94m{path}\033[0m")

    # --- Load dataset ---
    df = pd.read_csv(path)

    # Drop index column (first col) if redundant
    df = df.iloc[:, 1:] if df.columns[0].lower() in ["Unnamed: 0", "index"] else df

    # --- Clean dataset ---
    logger.info("Dropping duplicate rows..")
    cleaned_df = df.drop_duplicates()

    # --- Encode Suitable column for analysis ---
    _df = cleaned_df.copy()
    if "Suitable" in _df.columns:
        _df["Suitable"] = (
            _df["Suitable"].map({"Yes": 1, "No": 2}).fillna(_df["Suitable"])
        )

    # --- Key statistics grouped by Age ---
    if "ap_Age" in _df.columns:
        key_stats = _df.groupby("ap_Age").agg(
            {
                "ap_Education": ["mean", "min", "max", "std"],
                "ap_YearsOfExperience": ["mean", "min", "max", "std"],
                "ap_TechnicalScore": ["mean", "min", "max", "std"],
                "ap_InterviewScore": ["mean", "min", "max", "std"],
            }
        )
        key_stats.columns = [
            "_".join(col) for col in key_stats.columns
        ]  # flatten MultiIndex
        logger.info(
            "\033[93===================mKey Statistics by Age:=======================\033[0m"
        )
        print(key_stats)

    # --- Dataset overview ---
    logger.info("\033[93m============== Dataset Overview ==============\033[0m\n")
    logger.info(f"\033[93mShape (rows, cols):\033[0m {df.shape}\n")
    logger.info(
        f"\033[93m===================Columns:===================\033[0m\n {', '.join(df.columns)}\n"
    )
    logger.info(
        "\033[93m===================Missing values per column:===================\033[0m\n"
        + str(df.isnull().sum())
    )
    logger.info(
        "\033[93m=====Column data types:=================\033[0m\n" + str(df.dtypes)
    )

    return cleaned_df


if __name__ == "__main__":
    cleaned = inspect_dataset(open_dataset)
