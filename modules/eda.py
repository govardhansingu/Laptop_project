import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Display EDA charts from cleaned laptop dataset
def show_eda(df):
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()

    # Treat outliers using mean for numeric columns
    for col in numeric_cols:
        mean_val = df[col].mean()
        std_val = df[col].std()
        df[col] = df[col].apply(lambda x: mean_val if abs(x - mean_val) > 3 * std_val else x)

    st.subheader("💡 Price Distribution")
    fig, ax = plt.subplots()
    sns.histplot(df['Price'], kde=True, ax=ax, color="skyblue")
    ax.set_title("Distribution of Laptop Prices")
    st.pyplot(fig)

    st.subheader("📌 Relationships Between Features and Price")
    pairs = [('Company', 'Price'), ('Cpu_Processor', 'Price'), ('Ram', 'Price'), ('SSD', 'Price'), ('Touchscreen', 'Price'), ('PPI', 'Price')]

    for i in range(0, len(pairs), 3):
        cols = st.columns(3)
        for j in range(3):
            if i + j < len(pairs):
                x_col, y_col = pairs[i + j]
                with cols[j]:
                    fig, ax = plt.subplots()
                    sns.barplot(x=x_col, y=y_col, data=df, ax=ax, palette="viridis")
                    ax.set_title(f"{x_col} vs {y_col}")
                    plt.xticks(rotation=45)
                    st.pyplot(fig)

    st.subheader("📈 Boxplots for Numeric Features")
    for i in range(0, len(numeric_cols), 3):
        cols = st.columns(3)
        for j in range(3):
            if i + j < len(numeric_cols):
                col = numeric_cols[i + j]
                with cols[j]:
                    if df[col].dropna().empty:
                        st.warning(f"⚠️ Column '{col}' is empty or non-numeric.")
                        continue
                    fig, ax = plt.subplots()
                    sns.boxplot(y=df[col].dropna(), ax=ax, color="orange")
                    ax.set_title(f"Boxplot: {col}")
                    st.pyplot(fig)

    st.subheader("🔗 Scatter Plots: Features vs Price")
    scatter_pairs = [col for col in numeric_cols if col != "Price"]
    for i in range(0, len(scatter_pairs), 3):
        cols = st.columns(3)
        for j in range(3):
            if i + j < len(scatter_pairs):
                col = scatter_pairs[i + j]
                with cols[j]:
                    fig, ax = plt.subplots()
                    sns.scatterplot(x=col, y="Price", data=df, ax=ax, color="crimson")
                    ax.set_title(f"{col} vs Price")
                    st.pyplot(fig)

    st.subheader("📊 Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
    ax.set_title("Correlation Matrix")
    st.pyplot(fig)
    st.markdown("🔍 This heatmap shows the strength of relationships among numeric features.")
