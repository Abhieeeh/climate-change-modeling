import pandas as pd

def load_data(filepath):
    df = pd.read_csv(filepath)
    return df

def clean_data(df):
    df = df.dropna()
    return df
