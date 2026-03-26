import pandas as pd
import joblib
from tensorflow.keras.models import load_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from config import PROCESSED_PATH, MODEL_RF, MODEL_ANN, SCALER_PATH
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt



def evaluate():
    df = pd.read_csv(PROCESSED_PATH)

    X = df.drop("classification", axis=1)
    y = df["classification"]

    # split 
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y   # ✅ IMPORTANT0
    )


    # ---------------- RF ----------------
    rf = joblib.load(MODEL_RF)
    rf_pred = rf.predict(X_test)

    print("\n--- Random Forest ---")
    print(classification_report(y_test, rf_pred))

    # ---------------- ANN ----------------
    scaler = joblib.load(SCALER_PATH)
    X_test_scaled = scaler.transform(X_test)

    ann = load_model(MODEL_ANN)
    ann_pred = (ann.predict(X_test_scaled) > 0.5).astype(int)

    print("\n--- ANN ---")
    print(classification_report(y_test, ann_pred))

    cm = confusion_matrix(y_test, rf_pred)

    sns.heatmap(cm, annot=True, fmt='d')
    plt.title("Confusion Matrix - RF")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

if __name__ == "__main__":
    evaluate()