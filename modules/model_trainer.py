import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

@st.cache_data
def train_models(df, target_column="Price"):
    # Drop rows where target or features are missing
    df = df.dropna(subset=[target_column])
    df = df.dropna()

    # Prepare features and labels
    X = pd.get_dummies(df.drop(columns=[target_column]), drop_first=True)
    y = df[target_column]

    # Handle missing values
    imputer = SimpleImputer(strategy="mean")
    X_array = imputer.fit_transform(X)
    X = pd.DataFrame(X_array, columns=X.columns[:X_array.shape[1]])

    # Feature scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # Initialize models
    models = {
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(),
        "Random Forest": RandomForestRegressor(),
        "Gradient Boosting": GradientBoostingRegressor()
    }

    scores = {}
    best_model = None
    best_score = -1

    st.subheader("🔍 Model Performance (R² Scores)")
    for name, model in models.items():
        model.fit(X_train, y_train)
        score = r2_score(y_test, model.predict(X_test))
        scores[name] = score
        st.write(f"{name}: R² = {score:.2%}")
        if score > best_score:
            best_score = score
            best_model = model

    st.success(f"🏆 Best Model: {type(best_model).__name__} with R² = {best_score:.2%}")
    return best_model, X.columns, scaler, df
