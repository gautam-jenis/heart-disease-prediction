import numpy as np

def clean_zeros(data):
    data = data.copy()
    data[["Cholesterol", "RestingBP"]] = data[
        ["Cholesterol", "RestingBP"]
    ].replace(0, np.nan)
    return data