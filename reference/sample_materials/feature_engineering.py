def feature_engineering(X):
    """Drop 'test' column and create a new interaction feature 'BMI_Age_Interaction'."""
    X = X.copy()
    if 'test' in X.columns:
        X = X.drop(columns=['test'], errors='ignore')
    X["BMI_Age_Interaction"] = X["mass"] * X["age"]
    return X