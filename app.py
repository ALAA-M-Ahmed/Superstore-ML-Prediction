import streamlit as st
import pandas as pd
import joblib  # بدل pickle

st.set_page_config(page_title="Superstore ML Prediction", layout="centered")
st.title("📊 Superstore ML Dashboard")

# --- تحميل الموديلات والـ preprocessor ---
clf_model, clf_preprocessor = joblib.load("clf_model.joblib")
reg_model, reg_preprocessor = joblib.load("reg_model.joblib")
mc_model, mc_preprocessor = joblib.load("mc_model.joblib")

st.subheader("Enter Customer / Order Data")

# Numeric inputs
sales = st.number_input("Sales", min_value=0.0, value=100.0)
quantity = st.number_input("Quantity", min_value=0, value=1)
discount = st.number_input("Discount", min_value=0.0, max_value=1.0, value=0.1)
shipping_duration = st.number_input("Shipping Duration (days)", min_value=0, value=3)

# Categorical inputs
ship_mode = st.selectbox("Ship Mode", ["First Class", "Second Class", "Standard Class", "Same Day"])
segment = st.selectbox("Segment", ["Consumer", "Corporate", "Home Office"])
category = st.selectbox("Category", ["Furniture", "Office Supplies", "Technology"])
region = st.selectbox("Region", ["East", "West", "Central", "South"])
city = st.text_input("City", value="New York")

# --- تجهيز البيانات ---
input_df = pd.DataFrame({
    'Sales': [sales],
    'Quantity': [quantity],
    'Discount': [discount],
    'Shipping Duration': [shipping_duration],
    'Ship Mode': [ship_mode],
    'Segment': [segment],
    'Category': [category],
    'Region': [region],
    'City': [city]
})

if st.button("Predict"):
    # استخدم preprocessor قبل الموديل
    input_processed_clf = clf_preprocessor.transform(input_df)
    input_processed_reg = reg_preprocessor.transform(input_df)
    input_processed_mc = mc_preprocessor.transform(input_df)

    # Binary Classification - Loss Flag
    loss_flag = clf_model.predict(input_processed_clf)[0]
    loss_prob = clf_model.predict_proba(input_processed_clf)[:,1][0]

    # Regression - Predicted Profit
    pred_profit = reg_model.predict(input_processed_reg)[0]

    # Multi-class Classification - Profit Category
    profit_cat = mc_model.predict(input_processed_mc)[0]

    st.subheader("Predictions 🔮")
    st.write(f"🔹 Loss Flag: {'Yes' if loss_flag==1 else 'No'} (Probability: {loss_prob:.2f})")
    st.write(f"🔹 Predicted Profit: ${pred_profit:.2f}")
    st.write(f"🔹 Profit Category: {profit_cat}")
    st.success("✅ Prediction Completed!")

    st.subheader("Recommendations 💡")
    if loss_flag == 1:
        st.warning("⚠️ This order is likely to lose money. Consider adjusting pricing or discount.")
    else:
        st.info("✅ Order seems profitable.")

    if pred_profit < 50:
        st.info("💡 Profit is low. You can optimize shipping costs or discount strategy.")
    elif pred_profit < 200:
        st.info("💡 Profit is medium. Maintain current strategy.")
    else:
        st.success("🚀 Profit is high. Great performance!")

    if profit_cat == "Low":
        st.info("🔹 Profit category is Low: focus on improving efficiency and cost control.")
    elif profit_cat == "Medium":
        st.info("🔹 Profit category is Medium: monitor performance for optimization.")
    else:
        st.success("🔹 Profit category is High: keep scaling successful strategies.")
