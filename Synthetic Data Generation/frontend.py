import streamlit as st
import pandas as pd
import subprocess
import os
import uuid


if os.path.exists("augmented_dataset.csv"):
    os.remove("augmented_dataset.csv")
# ---------------------------------
# PAGE CONFIG
# ---------------------------------
st.set_page_config(
    page_title="CTGAN Data Augmentation",
    layout="centered"
)

st.title("🧬 CTGAN Synthetic Data Augmentation")
st.write(
    "Upload **any CSV dataset**. "
    "The backend CTGAN model will generate a synthetic-augmented dataset."
)

# ---------------------------------
# SETUP UPLOAD DIRECTORY
# ---------------------------------
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# ---------------------------------
# FILE UPLOADER
# ---------------------------------
uploaded_file = st.file_uploader(
    "📂 Upload CSV Dataset",
    type=["csv"]
)

if uploaded_file is not None:
    # Create unique filename
    unique_filename = f"{uuid.uuid4().hex}_{uploaded_file.name}"
    file_path = os.path.join(UPLOAD_DIR, unique_filename)

    # Save uploaded file
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.success(f"File uploaded successfully: **{uploaded_file.name}**")

    # Preview data
    try:
        df_preview = pd.read_csv(file_path)
        st.subheader("🔍 Dataset Preview")
        st.dataframe(df_preview.head())
        st.write(f"Rows: {df_preview.shape[0]} | Columns: {df_preview.shape[1]}")
    except Exception as e:
        st.error("Unable to read CSV file.")
        st.stop()

    # ---------------------------------
    # RUN BACKEND
    # ---------------------------------
    if st.button("🚀 Generate Augmented Dataset"):
        with st.spinner("Running CTGAN backend... This may take a few minutes ⏳"):

            import sys

# Use the same Python that runs Streamlit
            python_exe = sys.executable

            result = subprocess.run(
                [python_exe, "backend.py", file_path],
                capture_output=True,
                text=True
            )



        st.subheader("🖥 Backend Logs")
        #st.text(result.stdout)

        with st.expander("Show Backend Output (click to expand)", expanded=True):
            st.text_area(
                #label="",
                #value=result.stdout,
                #height=300  # scrollable box height

                label="Backend Logs",
                value=result.stdout if result.stdout else "No output from backend.",
                height=250,
                label_visibility="collapsed"
            )



        if result.returncode != 0:
            st.warning("Backend finished with warnings/errors")
            st.text(result.stderr)
        else:
            st.success("Augmented dataset generated successfully 🎉")

# ---------------------------------
# DOWNLOAD SECTION
# ---------------------------------
if os.path.exists("augmented_dataset.csv"):
    st.divider()
    st.subheader("⬇️ Download Augmented Dataset")

    augmented_df = pd.read_csv("augmented_dataset.csv")
    st.dataframe(augmented_df.head())
    st.write(f"Rows: {augmented_df.shape[0]} | Columns: {augmented_df.shape[1]}")
    ######above

    st.download_button(
        label="Download augmented_dataset.csv",
        data=augmented_df.to_csv(index=False).encode("utf-8"),
        file_name="augmented_dataset.csv",
        mime="text/csv"
    )

    st.info(
        "The augmented dataset contains both original and CTGAN-generated synthetic data."
    )
