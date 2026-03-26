import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib
from config import PROCESSED_PATH, MODEL_RF
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split


def train_rf():
    df = pd.read_csv(PROCESSED_PATH)

    X = df.drop("classification", axis=1)
    y = df["classification"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y   # changed from None to y
    )


    model = RandomForestClassifier(n_estimators=200)
    model.fit(X_train, y_train)

    scores = cross_val_score(model, X, y, cv=5)

    print("Cross-validation accuracy:", scores)
    print("Mean accuracy:", scores.mean())

    joblib.dump(model, MODEL_RF)
    print("Random Forest trained!")

if __name__ == "__main__":
    train_rf()