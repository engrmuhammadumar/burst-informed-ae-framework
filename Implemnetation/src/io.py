from pathlib import Path
import numpy as np
import pandas as pd

def load_index(csv_path, label_col="label"):
    df = pd.read_csv(csv_path)
    assert label_col in df.columns, f"{label_col} not in CSV"
    return df

def load_signal(path, file_type="npy"):
    p = Path(path)
    if file_type == "npy":
        return np.load(p)
    elif file_type == "csv":
        return np.loadtxt(p, delimiter=",")
    else:
        raise ValueError("file_type must be 'npy' or 'csv'")
