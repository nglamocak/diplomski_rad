"""Data processing."""
import pandas as pd
from pandas import DataFrame, Series


def clean_data(path: str = "data.csv") -> DataFrame:
    """Cleans the data from the given CSV file path."""
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()

    df["Income"] = pd.to_numeric(df["Income"], errors="coerce")
    df["Age"] = pd.to_numeric(df["Age"], errors="coerce")

    df = df[df['Income'] != 35000]

    df["Marital status"] = df["Marital status"].replace(
        {"unmarred": "unmarried", "unmaried": "unmarried"})

    for col in ["Education", "Employment", "Marital status", "Violence"]:
        if col in df.columns:
            df[col] = (
                df[col]
                .astype(str)
                .str.strip()
                .str.lower()
                .str.replace(r"\s+", " ", regex=True)
            )

    df["Employment"] = df["Employment"].replace(
        {
            "employed ": "employed",
            "semi-employed": "semi employed",
        }
    )

    def categorize_income(x: object) -> str:
        try:
            x = float(x)  # type: ignore[arg-type]
        except (ValueError, TypeError):
            return "unknown"
        if x == 0:
            return "no_income"
        elif 0 < x <= 500:
            return "very_low"
        elif 500 < x <= 2000:
            return "low"
        elif 2000 < x <= 5000:
            return "middle"
        elif 5000 < x <= 10000:
            return "upper_middle"
        else:
            return "high"

    if "Income" in df.columns:
        df["Income"] = df["Income"].apply(categorize_income)
    else:
        df["Income"] = "unknown"

    df[["Education", "Employment", "Marital status", "Income"]] = df[
        ["Education", "Employment", "Marital status", "Income"]
    ].fillna("unknown")

    return df


def calculate_risk_score(row: Series) -> int:
    """Calculates the risk score."""
    score = 0
    try:
        age = int(row["Age"])
    except (ValueError, TypeError):
        age = -1

    if age < 20:
        score += 0
    elif 20 <= age <= 29:
        score += 1
    elif 30 <= age <= 39:
        score += 2
    elif 40 <= age <= 49:
        score += 3
    elif age >= 50:
        score += 2

    # Education
    education_points = {
        'primary': 2,
        'none': 1,
        'tertiary': 1,
        'secondary': 0,
        'unknown': 0,
    }
    score += education_points.get(str(row["Education"]).lower(), 0)

    # Employment
    employment_points = {
        'unemployed': 3,
        'semi employed': 1,
        'employed': 0,
        'unknown': 0,
    }
    score += employment_points.get(str(row["Employment"]).lower(), 0)

    # Marital
    marital_status_points = {
        'married': 3,
        'unmarried': 0,
        'unknown': 0,
    }
    score += marital_status_points.get(str(row["Marital status"]).lower(), 0)

    # Income
    income_points = {
        'no_income': 3,
        'very_low': 2,
        'low': 1,
        'middle': -1,
        'upper_middle': -2,
        'high': -3,
        'unknown': 0,
    }
    score += income_points.get(str(row["Income"]).lower(), 0)

    inc = str(row["Income"]).lower()
    mar = str(row["Marital status"]).lower()
    if mar == "married" and inc == "no_income":
        score += 2

    return score


def calculate_weight(row: Series) -> int:
    """Calculates the weight."""
    weight = 0
    try:
        age = int(row["Age"])
    except (ValueError, TypeError):
        age = -1

    # Dob
    if 16 <= age < 25:
        weight += 1
    elif 25 <= age <= 55:
        weight += 2
    elif age > 55:
        weight += 2

    # Education
    education_weight = {
        'none': 1,
        'primary': 1,
        'secondary': 1,
        'tertiary': 0,
        'unknown': 0,
    }
    weight += education_weight.get(str(row["Education"]).lower(), 0)

    # Employment
    employment_weight = {
        'unemployed': 2,
        'semi employed': 1,
        'employed': 0,
        'unknown': 0,
    }
    weight += employment_weight.get(str(row["Employment"]).lower(), 0)

    # Marital
    marital_status_weight = {
        'married': 1,
        'unmarried': 0,
        'unknown': 0,
    }
    weight += marital_status_weight.get(str(row["Marital status"]).lower(), 0)

    # Income
    income_weight = {
        'no_income': 2,
        'very_low': 2,
        'low': 1,
        'middle': 0,
        'upper_middle': 0,
        'high': 0,
        'unknown': 1,
    }
    weight += income_weight.get(str(row["Income"]).lower(), 1)

    inc = str(row["Income"]).lower()
    mar = str(row["Marital status"]).lower()
    if mar == "married" and inc == "no_income":
        weight -= 1

    return max(1, weight)


def prepare_dataset(path: str) -> pd.DataFrame:
    """Prepares the dataset by cleaning and adding risk_score and weight."""
    df = clean_data(path)
    df["risk_score"] = df.apply(calculate_risk_score, axis=1)
    df["weight"] = df.apply(calculate_weight, axis=1)

    return df
