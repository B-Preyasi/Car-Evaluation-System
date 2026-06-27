import streamlit as st
import os

from src.predict import predict_car


st.set_page_config(
    page_title="Car Evaluation System",
    page_icon="🚗",
    layout="wide"
)

st.sidebar.title("🚗 Car Evaluation System")
st.sidebar.markdown("""
This system predicts car acceptability using:

- Buying price
- Maintenance cost
- Number of doors
- Seating capacity
- Luggage boot size
- Safety level
""")



st.title("🚗 Car Evaluation System")
st.markdown("### Predicts Car Acceptability ")

st.markdown("""
This application predicts whether a car is:

- 🚫 **Unacceptable**
- ✅ **Acceptable**
- 👍 **Good**
- 🌟 **Very Good**
""")

st.divider()

left_col, right_col = st.columns([1, 1])

with left_col:
    st.subheader("📝 Enter Car Details")

    buying = st.selectbox("Buying Price", ["low", "med", "high", "vhigh"])
    maint = st.selectbox("Maintenance Cost", ["low", "med", "high", "vhigh"])
    doors = st.selectbox("Number of Doors", ["2", "3", "4", "5more"])
    persons = st.selectbox("Seating Capacity", ["2", "4", "more"])
    lug_boot = st.selectbox("Luggage Boot Size", ["small", "med", "big"])
    safety = st.selectbox("Safety Level", ["low", "med", "high"])

    predict_button = st.button("🚀 Evaluate Car", use_container_width=True)

with right_col:
    st.subheader("🔍 Prediction Dashboard")

    if predict_button:
        result, confidence = predict_car(
            buying, maint, doors, persons, lug_boot, safety
        )

        st.metric("Predicted Class", result.upper())
        st.metric("Confidence Score", f"{confidence:.2f}%")

        if result == "unacc":
            st.error("🚫 This car is Unacceptable")
            st.warning("Recommendation: Avoid this car. It may have poor safety, high cost, or low capacity.")

        elif result == "acc":
            st.info("✅ This car is Acceptable")
            st.write("Recommendation: This car is usable, but there may be better options.")

        elif result == "good":
            st.success("👍 This car is Good")
            st.write("Recommendation: This car has a balanced combination of price, comfort, and safety.")

        elif result == "vgood":
            st.success("🌟 This car is Very Good")
            st.balloons()
            st.write("Recommendation: This car is highly recommended.")

    else:
        st.info("Select car details and click Evaluate Car.")

st.divider()

st.subheader("📊 Model Evaluation Visuals")

col1, col2 = st.columns(2)

with col1:
    if os.path.exists("outputs/confusion_matrix.png"):
        st.image("outputs/confusion_matrix.png", caption="Confusion Matrix")
    else:
        st.warning("Run `python -m src.evaluate_model` to generate confusion matrix.")

with col2:
    if os.path.exists("outputs/feature_importance.png"):
        st.image("outputs/feature_importance.png", caption="Feature Importance")
    else:
        st.warning("Run `python -m src.evaluate_model` to generate feature importance chart.")

st.divider()

