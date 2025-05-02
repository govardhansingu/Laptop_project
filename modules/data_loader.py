import pandas as pd
import os

def load_laptop_data(path="E:/Laptop_project/laptop.csv"):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found at: {path}")
    return pd.read_csv(path)
