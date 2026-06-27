import pandas as pd
from sklearn.preprocessing import LabelEncoder

def load_and_preprocess_data():
    columns=["buying","maint","doors","persons","lug_boot","safety","class"]

    df=pd.read_csv("data/car.data",names=columns)
    print("First 5 rows:")
    print(df.head())

    encoders={}

    for column in df.columns:
        encoder=LabelEncoder()
        df[column]=encoder.fit_transform(df[column])
        encoders[column]=encoder

    X = df.drop("class", axis=1)
    y = df["class"]

    return X, y, encoders    