import streamlit as st
import pandas as pd
from modules.data_loader import load_laptop_data
from modules.cleaning import clean_laptop_data
from modules.eda import show_eda
from modules.model_trainer import train_models
import io

st.set_page_config(page_title="Laptop Price Predictor", layout="wide")
st.title("💻 Laptop Price Prediction App")

# Session state
if "df_cleaned" not in st.session_state:
    st.session_state.df_cleaned = None
if "model" not in st.session_state:
    st.session_state.model = None
if "features" not in st.session_state:
    st.session_state.features = None
if "scaler" not in st.session_state:
    st.session_state.scaler = None

# Load and clean data
st.header("📥 Load Dataset")
if st.button("Load and Clean Data"):
    try:
        df_raw = load_laptop_data()
        df_cleaned = clean_laptop_data(df_raw)
        st.session_state.df_cleaned = df_cleaned
        st.success("✅ Data loaded and cleaned successfully!")
        st.dataframe(df_cleaned.head())

        # Download link
        buffer = io.BytesIO()
        df_cleaned.to_csv(buffer, index=False)
        st.download_button("📂 Download Cleaned Dataset", buffer, file_name="laptop_clean_data.csv", mime="text/csv")

    except Exception as e:
        st.error(f"❌ Error: {e}")

# EDA
if st.session_state.df_cleaned is not None:
    if st.checkbox("📊 Show EDA Analysis"):
        show_eda(st.session_state.df_cleaned)

# Model training
if st.session_state.df_cleaned is not None:
    if st.button("🤖 Train Models"):
        model, feature_cols, scaler, df_final = train_models(st.session_state.df_cleaned)
        st.session_state.model = model
        st.session_state.features = feature_cols
        st.session_state.scaler = scaler
        st.session_state.df_final = df_final

# Prediction
if st.session_state.model is not None:
    st.header("🔮 Predict Laptop Price")

    df_ref = st.session_state.df_cleaned
    feature_cols = st.session_state.features

    st.subheader("📥 Enter Laptop Features to Predict Price")

    # Define original input columns used before one-hot encoding
    input_columns = ['Company', 'Typename', 'Ram', 'Weight', 'Touchscreen', 'Ips', 'Inches',
                     'Cpu_Processor', 'Gpu_brand', 'HDD', 'SSD', 'OpSys', 'ppi']

    input_data_raw = {}
    for col in input_columns:
        if col in df_ref.columns:
            unique_values = df_ref[col].dropna().unique()
            input_data_raw[col] = st.selectbox(f"{col}", sorted(unique_values))

    if st.button("💡 Predict Price"):
        try:
            # Convert raw inputs to DataFrame
            input_df = pd.DataFrame([input_data_raw])

            # One-hot encode
            input_encoded = pd.get_dummies(input_df)
            input_encoded = input_encoded.reindex(columns=feature_cols, fill_value=0)

            # Scale input
            scaled_input = st.session_state.scaler.transform(input_encoded)
            prediction = st.session_state.model.predict(scaled_input)[0]
            st.success(f"💰 Predicted Laptop Price: ₹{prediction:,.2f}")
        except Exception as e:
            st.error(f"⚠️ Prediction failed: {e}")
