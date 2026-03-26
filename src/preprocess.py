import pandas as pd
import numpy as np
from config import DATA_PATH, PROCESSED_PATH

def preprocess():
    df = pd.read_csv(DATA_PATH)

    if 'id' in df.columns:
        df.drop('id', axis=1, inplace=True)

    df.replace('?', np.nan, inplace=True)

    df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)

    numeric_cols = ['age','bp','sg','al','su','bgr','bu','sc','sod','pot',
                    'hemo','pcv','wc','rc']

    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df['classification'] = df['classification'].map({
        'ckd': 1,
        'notckd': 0
    })

    # Fill missing values
    for col in df.columns:
        if col != 'classification':
            if df[col].dtype == 'object':
                df[col] = df[col].fillna(df[col].mode()[0])
            else:
                df[col] = df[col].fillna(df[col].mean())

    # Encode categorical 
    df = pd.get_dummies(df, drop_first=True)

    cols = [c for c in df.columns if c != 'classification'] + ['classification']
    df = df[cols]

    df.to_csv(PROCESSED_PATH, index=False)
    print("✅ Data preprocessing completed!")

if __name__ == "__main__":
    preprocess()