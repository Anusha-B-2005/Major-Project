import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import tempfile
import os
import sys

# Add backend folder to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gain  # your backend module

st.set_page_config("GAIN Missing Data Imputation", layout="wide")
st.title("🧠 Missing Data Imputation (GAIN)")

uploaded_file = st.file_uploader("📂 Upload CSV file", type=["csv"])

if uploaded_file:
    original_df = pd.read_csv(uploaded_file)
    st.subheader("Uploaded Dataset")
    st.dataframe(original_df.head())

    st.subheader("📍 Missing Values Summary")
    missing_summary = pd.DataFrame(original_df.isnull().sum(), columns=["Missing Values"])
    st.markdown("<div style='text-align:center'>", unsafe_allow_html=True)
    st.table(missing_summary)
    st.markdown("</div>", unsafe_allow_html=True)


    st.subheader("📉 Missing Values Visualization")
    fig, ax = plt.subplots()
    original_df.isnull().sum().plot(kind="bar", ax=ax)
    st.pyplot(fig)


    if st.button("🚀 Run GAIN Imputation"):
        with st.spinner("Running GAIN… Please wait"):

            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp_file:
                tmp_file.write(uploaded_file.getbuffer())
                temp_path = tmp_file.name

            # =========================
            # Backend: Load + Prepare
            # =========================
            df_original, df_ohe, categorical_columns = gain.load_and_ohe(temp_path)
            X_train, M_train, scaler, scaled_data, eval_mask, column_names = gain.prepare_data_for_gain(df_ohe)

            generator, discriminator = gain.build_gain(X_train.shape[1])
            gain.train_gain(generator, discriminator, X_train, M_train)

            # Impute missing values
            imputed_df = gain.impute(generator, X_train, M_train, scaler, column_names, categorical_columns)

            # =========================
            # Restore Categorical Columns Safely
            # =========================
            df_final = imputed_df.copy()

            for cat_col in categorical_columns:
                one_hot_cols = [c for c in df_final.columns if c.startswith(cat_col + "_")]
                if not one_hot_cols:
                    continue

                # Convert one-hot to single label
                restored = df_final[one_hot_cols].idxmax(axis=1).str.replace(cat_col + "_", "", regex=False)
                restored = restored.where(original_df[cat_col].isna(), original_df[cat_col])
                df_final[cat_col] = restored
                df_final.drop(columns=one_hot_cols, inplace=True)

            # =========================
            # Numeric columns: preserve original non-missing
            # =========================
            for col in original_df.columns:
                df_final[col] = original_df[col].combine_first(df_final[col])

            # Clean up temp file
            os.remove(temp_path)

        st.success("✅ Missing values imputed successfully")

        st.subheader("Final Dataset (Original Values Preserved)")
        st.dataframe(df_final.head())

        st.subheader("📍 Missing Values Summary After GAIN")
        missing_after = pd.DataFrame(df_final.isnull().sum(), columns=["Missing Values"])
        st.markdown("<div style='text-align:center'>", unsafe_allow_html=True)
        st.table(missing_after)
        st.markdown("</div>", unsafe_allow_html=True)

        st.download_button(
            "⬇️ Download Final CSV",
            df_final.to_csv(index=False),
            "final_imputed_dataset.csv",
            "text/csv"
        )
