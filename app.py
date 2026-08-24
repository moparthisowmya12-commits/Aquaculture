import streamlit as st
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier


st.set_page_config(page_title="Aquaculture & Livestock Disease Risk", page_icon="🐟", layout="wide")

st.title("🐟 Aquaculture & Livestock Disease Risk Prediction")
st.write("Upload the CSV dataset, train the model, and predict the disease-risk level.")

# Default dataset filename
DEFAULT_FILE = "aquaculture_livestock_disease_feed_optimization_dataset(5).csv"

uploaded_file = st.file_uploader("Upload your CSV dataset", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
else:
    try:
        df = pd.read_csv(DEFAULT_FILE)
    except FileNotFoundError:
        st.info(f"Please upload `{DEFAULT_FILE}` using the uploader above.")
        st.stop()

st.success(f"Dataset loaded: {df.shape[0]} rows × {df.shape[1]} columns")

target = "Disease_Risk_Level"

if target not in df.columns:
    st.error(f"Target column `{target}` was not found in the dataset.")
    st.stop()

# Remove ID column because it is not useful for prediction
drop_cols = [c for c in ["Farm_ID", target] if c in df.columns]
X = df.drop(columns=drop_cols)
y = df[target].astype(str)

# Separate numeric and categorical columns
numeric_cols = X.select_dtypes(include=["number"]).columns.tolist()
categorical_cols = X.select_dtypes(exclude=["number"]).columns.tolist()

preprocessor = ColumnTransformer(
    transformers=[
        ("num", "passthrough", numeric_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
    ]
)

model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(
            n_estimators=300,
            random_state=42,
            class_weight="balanced"
        )),
    ]
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

with st.spinner("Training model..."):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

st.metric("Model Accuracy", f"{accuracy * 100:.2f}%")

st.divider()
st.subheader("🔮 Predict Disease Risk")

# Create input widgets from the dataset columns
input_data = {}

for col in X.columns:
    if col in categorical_cols:
        values = sorted(X[col].dropna().astype(str).unique().tolist())
        input_data[col] = st.selectbox(col, values)
    else:
        value = float(X[col].median())
        min_value = float(X[col].min())
        max_value = float(X[col].max())

        if min_value == max_value:
            input_data[col] = value
        else:
            input_data[col] = st.number_input(
                col,
                min_value=min_value,
                max_value=max_value,
                value=value
            )

if st.button("Predict Disease Risk", type="primary"):
    input_df = pd.DataFrame([input_data])

    try:
        prediction = model.predict(input_df)[0]
        probabilities = model.predict_proba(input_df)[0]
        classes = model.named_steps["classifier"].classes_

        st.success(f"Predicted Disease Risk Level: **{prediction}**")

        probability_df = pd.DataFrame({
            "Risk Level": classes,
            "Probability": probabilities
        }).sort_values("Probability", ascending=False)

        st.dataframe(
            probability_df.style.format({"Probability": "{:.2%}"}),
            use_container_width=True
        )

    except Exception as e:
        st.error(f"Prediction error: {e}")

st.divider()
st.subheader("📊 Dataset Preview")
st.dataframe(df.head(20), use_container_width=True)

# Download the dataset from the Streamlit app
csv_data = df.to_csv(index=False).encode("utf-8")
st.download_button(
    label="⬇️ Download Dataset",
    data=csv_data,
    file_name="aquaculture_livestock_disease_feed_optimization_dataset.csv",
    mime="text/csv"
)
