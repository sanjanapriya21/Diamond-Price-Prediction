
"""Streamlit app for diamond price prediction with a KNN regression pipeline."""

import pickle
from pathlib import Path

import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "diamonds.csv"
MODEL_PATH = ROOT / "models" / "diamond_knn_pipeline.pkl"

TARGET = "price"
K_NEIGHBORS = 7

REQUIRED_FEATURES = (
    "carat",
    "cut",
    "color",
    "clarity",
    "depth",
    "table",
    "x",
    "y",
    "z",
)


st.set_page_config(
    page_title="Diamond Price Studio",
    page_icon="💎",
    layout="wide",
)


@st.cache_data(show_spinner=False)
def load_dataset(path: str) -> pd.DataFrame:
    """Load and validate the diamond dataset."""

    data = pd.read_csv(path)

    if TARGET not in data.columns:
        raise ValueError(f"The dataset must contain a '{TARGET}' column.")

    data = data.loc[:, ~data.columns.astype(str).str.match(r"^Unnamed:")]
    data = data.dropna(subset=[TARGET]).copy()

    if data.empty:
        raise ValueError("The dataset has no valid price values.")

    missing = [
        column
        for column in REQUIRED_FEATURES
        if column not in data.columns
    ]

    if missing:
        raise ValueError(
            f"The dataset is missing required features: {', '.join(missing)}."
        )

    return data


@st.cache_resource(show_spinner="Training the KNN model...")
def train_model(
    path: str,
    modified_at: float,
) -> tuple[Pipeline, pd.DataFrame, float, float, float]:
    """Train, evaluate and save the KNN pipeline."""

    del modified_at

    data = load_dataset(path)

    features = data.drop(columns=[TARGET])
    target = data[TARGET]

    numeric_features = features.select_dtypes(
        include=["number"]
    ).columns.tolist()

    categorical_features = features.select_dtypes(
        include=["object", "category", "string"]
    ).columns.tolist()

    # Preprocessing for numerical columns
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    # Preprocessing for categorical columns
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    # Combine both preprocessing pipelines
    preprocessing = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_features),
            ("categorical", categorical_pipeline, categorical_features),
        ]
    )

    # Create the complete ML pipeline
    pipeline = Pipeline(
        steps=[
            ("preprocessing", preprocessing),
            (
                "regressor",
                KNeighborsRegressor(
                    n_neighbors=K_NEIGHBORS,
                    weights="distance",
                ),
            ),
        ]
    )

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=0.25,
        random_state=42,
    )

    # Train the model
    pipeline.fit(X_train, y_train)

    # Predict prices using test data
    y_pred = pipeline.predict(X_test)

    # Calculate evaluation metrics
    mae = mean_absolute_error(y_test, y_pred)

    rmse = mean_squared_error(y_test, y_pred) ** 0.5

    r2 = r2_score(y_test, y_pred)

    # Save the trained pipeline
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    with MODEL_PATH.open("wb") as model_file:
        pickle.dump(pipeline, model_file)

    return pipeline, data, mae, rmse, r2


def main() -> None:
    """Run the Streamlit application."""

    if not DATA_PATH.is_file():
        st.error(f"Could not find the dataset: {DATA_PATH}")
        st.info("Place diamonds.csv in the same folder as daimond.py.")
        st.stop()

    try:
        pipeline, data, mae, rmse, r2 = train_model(
            str(DATA_PATH),
            DATA_PATH.stat().st_mtime,
        )

    except (OSError, ValueError, pd.errors.ParserError) as error:
        st.error(f"Unable to prepare the diamond model: {error}")
        st.stop()

    # Application heading
    st.markdown(
        """
        <div style="
            padding: 1.5rem;
            border-left: 5px solid #168477;
            border-radius: 10px;
            background: #172433;
            margin-bottom: 20px;
        ">
            <h1 style="color: white;">
                💎 Diamond Price Studio
            </h1>
            <p style="color: #cbd5e1;">
                Estimate a diamond's market price using a KNN
                regression model.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Display evaluation metrics
    st.subheader("Model Evaluation")

    col1, col2, col3 = st.columns(3)

    col1.metric("MAE", f"{mae:,.2f}")
    col2.metric("RMSE", f"{rmse:,.2f}")
    col3.metric("R² Score", f"{r2:.4f}")

    st.divider()

    # Prediction section
    st.subheader("Price Estimator")

    st.write(
        "Enter the diamond's characteristics to estimate its price."
    )

    with st.form("diamond_prediction_form"):

        first_row = st.columns(3)

        carat = first_row[0].number_input(
            "Carat",
            min_value=0.01,
            value=0.50,
            step=0.01,
            format="%.2f",
        )

        cut = first_row[1].selectbox(
            "Cut",
            options=sorted(data["cut"].dropna().unique().tolist()),
        )

        color = first_row[2].selectbox(
            "Color",
            options=sorted(data["color"].dropna().unique().tolist()),
        )

        second_row = st.columns(3)

        clarity = second_row[0].selectbox(
            "Clarity",
            options=sorted(data["clarity"].dropna().unique().tolist()),
        )

        depth = second_row[1].number_input(
            "Depth (%)",
            min_value=0.0,
            value=61.5,
            step=0.1,
            format="%.1f",
        )

        table = second_row[2].number_input(
            "Table (%)",
            min_value=0.0,
            value=57.0,
            step=0.1,
            format="%.1f",
        )

        third_row = st.columns(3)

        x = third_row[0].number_input(
            "Length (x, mm)",
            min_value=0.0,
            value=5.0,
            step=0.01,
            format="%.2f",
        )

        y = third_row[1].number_input(
            "Width (y, mm)",
            min_value=0.0,
            value=5.0,
            step=0.01,
            format="%.2f",
        )

        z = third_row[2].number_input(
            "Height (z, mm)",
            min_value=0.0,
            value=3.1,
            step=0.01,
            format="%.2f",
        )

        submitted = st.form_submit_button(
            "Estimate Price",
            type="primary",
            use_container_width=True,
        )

    # Predict diamond price
    if submitted:

        sample = pd.DataFrame(
            [
                {
                    "carat": carat,
                    "cut": cut,
                    "color": color,
                    "clarity": clarity,
                    "depth": depth,
                    "table": table,
                    "x": x,
                    "y": y,
                    "z": z,
                }
            ]
        )

        estimate = float(pipeline.predict(sample)[0])

        st.success("Prediction completed!")

        st.metric(
            "Estimated Diamond Price",
            f"${estimate:,.2f}",
        )

        st.caption(
            "This is a model estimate based on the training data, "
            "not a formal appraisal."
        )


if __name__ == "__main__":
    main()
