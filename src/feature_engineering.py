def create_rfm_features(df):

    if "Recency" in df.columns:
        df["R"] = df["Recency"]

    if "Frequency" in df.columns:
        df["F"] = df["Frequency"]

    if "Monetary" in df.columns:
        df["M"] = df["Monetary"]

    return df


def create_derived_features(df):

    if "Income" in df.columns and "SpendingScore" in df.columns:
        df["Income_Spending_Ratio"] = df["Income"] / (df["SpendingScore"] + 1)

    return df