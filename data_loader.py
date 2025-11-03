# data_loader.py
import pandas as pd

def load_dataset(path):
    """Memuat dataset dari path yang diberikan"""
    df = pd.read_csv(path)
    print("=== Dataset Loaded ===")
    print("Jumlah data:", len(df))
    print("Kolom:", df.columns.tolist())
    return df
