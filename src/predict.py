import joblib
import pandas as pd

model = joblib.load("models/car_model.pkl")
encoders = joblib.load("models/encoders.pkl")


def predict_car(buying, maint, doors, persons, lug_boot, safety):
    input_data = pd.DataFrame(
        [[buying, maint, doors, persons, lug_boot, safety]],
        columns=["buying", "maint", "doors", "persons", "lug_boot", "safety"]
    )

    for column in input_data.columns:
        input_data[column] = encoders[column].transform(input_data[column])

    prediction = model.predict(input_data)[0]
    probabilities = model.predict_proba(input_data)[0]

    result = encoders["class"].inverse_transform([prediction])[0]
    confidence = max(probabilities) * 100

    return result, confidence