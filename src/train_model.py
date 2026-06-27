import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

from src.data_preprocessing import load_and_preprocess_data


def train_model():
    X, y, encoders = load_and_preprocess_data()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    os.makedirs("models", exist_ok=True)

    joblib.dump(model, "models/car_model.pkl")
    joblib.dump(encoders, "models/encoders.pkl")
    joblib.dump(X.columns.tolist(), "models/features.pkl")

    print("Model trained successfully!")
    print(f"Accuracy: {accuracy:.2f}")
    print("Model, encoders, and features saved.")


if __name__ == "__main__":
    train_model()