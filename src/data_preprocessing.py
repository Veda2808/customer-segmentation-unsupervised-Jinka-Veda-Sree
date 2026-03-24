import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler


def handle_missing_values(df):
    """
    Fill missing numeric values with median
    """
    df = df.fillna(df.median(numeric_only=True))
    return df


def encode_categorical(df):
    """
    Encode categorical columns using LabelEncoder
    """
    le = LabelEncoder()

    for col in df.select_dtypes(include=['object']).columns:
        df[col] = le.fit_transform(df[col])

    return df


def remove_outliers(df):
    """
    Remove outliers using IQR method
    """
    Q1 = df.quantile(0.25)
    Q3 = df.quantile(0.75)

    IQR = Q3 - Q1

    df = df[~((df < (Q1 - 1.5 * IQR)) |
              (df > (Q3 + 1.5 * IQR))).any(axis=1)]

    return df


def scale_features(df):
    """
    Standardize features
    """
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df)

    return scaled_data


def preprocess_data(df):
    """
    Complete preprocessing pipeline
    """

    # Handle missing values
    df = handle_missing_values(df)

    # Encode categorical variables
    df = encode_categorical(df)

    # Remove outliers
    df = remove_outliers(df)

    return df