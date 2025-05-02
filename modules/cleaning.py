import pandas as pd
import numpy as np

def clean_laptop_data(df):
    # Replace '?' with NaN
    df.replace('?', np.nan, inplace=True)

    # Drop unnamed columns and rows with missing data
    df.drop(columns=[col for col in df.columns if "Unnamed" in col], inplace=True)
    df.dropna(inplace=True)

    # Clean and extract features
    df["Company"] = df["Company"].str.strip()
    df["Touchscreen"] = df["ScreenResolution"].apply(lambda x: 1 if 'Touchscreen' in x else 0)
    df["IPS"] = df["ScreenResolution"].apply(lambda x: 1 if 'IPS' in x else 0)

    df["Inches"] = pd.to_numeric(df["Inches"], errors='coerce')
    res = df["ScreenResolution"].str.extract(r'(\d+)x(\d+)')
    df["x_res"] = pd.to_numeric(res[0], errors='coerce')
    df["y_res"] = pd.to_numeric(res[1], errors='coerce')
    df["PPI"] = ((df["x_res"]**2 + df["y_res"]**2)**0.5) / df["Inches"]

    def find_processor_type(x):
        processor = " ".join(x.split()[0:3])
        if processor in ['Intel Core i7', 'Intel Core i5', 'Intel Core i3']:
            return processor
        elif processor.startswith('Intel'):
            return 'Other Intel Processor'
        elif processor.startswith('AMD'):
            return 'AMD Processor'
        else:
            return "Other_Processor"
    df['Cpu_Processor'] = df['Cpu'].apply(find_processor_type)

    df["Ram"] = df["Ram"].str.replace("GB", "").astype(int)
    df["Weight"] = df["Weight"].str.replace("kg", "").astype(float)

    df["Memory"] = df["Memory"].astype(str).replace(".0", "", regex=True).str.replace("GB", "").str.replace("TB", "000")
    new = df["Memory"].str.split("+", n=1, expand=True)
    df["first"] = new[0].str.extract(r'(\d+)').fillna(0).astype(int)
    df["second"] = new[1].str.extract(r'(\d+)').fillna(0).astype(int)

    df["HDD"] = df["first"]
    df["SSD"] = df["second"]

    df["Gpu_brand"] = df["Gpu"].apply(lambda x: x.split()[0])

    def OS_type(x):
        if x in ['Windows 10', 'Windows 7', 'Windows 10 S']:
            return 'Windows'
        elif x in ['macOS', 'Mac OS X']:
            return 'Mac'
        elif x == 'Linux':
            return 'Linux'
        else:
            return 'Others/No OS/Chrome'
    df['OS'] = df['OpSys'].apply(OS_type)

    df.drop(columns=['ScreenResolution', 'Cpu', 'Memory', 'first', 'second', 'x_res', 'y_res', 'Gpu', 'OpSys'], inplace=True)
    df.dropna(inplace=True)

    # Treat outliers (3-sigma rule)
    for col in df.select_dtypes(include=[np.number]):
        mean = df[col].mean()
        std = df[col].std()
        df[col] = df[col].apply(lambda x: mean if abs(x - mean) > 3 * std else x)

    return df
