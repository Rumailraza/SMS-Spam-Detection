import pandas as pd

def load_data(path):
    df = pd.read_csv(path)

    df['label'] = df['label'].map({
        'ham':0,
        'spam':1
    })

    return df