import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import io

st.set_page_config(page_title="CRISP-DM Linear Regression", layout="wide")

st.title("Linear Regression with CRISP-DM Methodology")
st.markdown("This app demonstrates the six phases of the Cross-Industry Standard Process for Data Mining (CRISP-DM) using a simple Linear Regression model on synthetic data.")

# --- SIDEBAR ---
st.sidebar.header("Data Generation Settings")
n = st.sidebar.slider("Number of samples (n)", 100, 1000, 500)
variance = st.sidebar.slider("Noise Variance", 0, 1000, 200)
seed = st.sidebar.slider("Random Seed", 1, 1000, 42)

generate_btn = st.sidebar.button("Generate Data")

@st.cache_data
def get_synthetic_data(n_samples, noise_var, random_seed):
    np.random.seed(random_seed)
    X = np.random.uniform(-100, 100, n_samples)
    a_true = np.random.uniform(-10, 10)
    b_true = np.random.uniform(-50, 50)
    noise_mean = np.random.uniform(-10, 10)
    noise = np.random.normal(noise_mean, np.sqrt(noise_var), n_samples)
    y = a_true * X + b_true + noise
    df = pd.DataFrame({'X': X, 'y': y})
    return df, a_true, b_true

# --- 1. Business Understanding ---
st.header("1. Business Understanding")
st.write("""
**Objective:** The goal is to build a predictive model that can accurately estimate a continuous target variable (`y`) based on a single feature (`X`).

**Success Criteria:** A high R² score and low Root Mean Squared Error (RMSE) indicating that the model generalizes well to unseen data.
""")

# --- Initialize data if button clicked or already in session state ---
if generate_btn:
    st.session_state['current_params'] = {'n': n, 'variance': variance, 'seed': seed}
    st.session_state['generated'] = True

if st.session_state.get('generated', False):
    params = st.session_state['current_params']
    df, a_true, b_true = get_synthetic_data(params['n'], params['variance'], params['seed'])
    
    # --- 2. Data Understanding ---
    st.header("2. Data Understanding")
    st.write("We have generated a synthetic dataset with a single feature (`X`) and a target variable (`y`).")
    col1, col2 = st.columns(2)
    with col1:
        st.write("Data Sample (First 5 rows):")
        st.dataframe(df.head(), use_container_width=True)
    with col2:
        st.write("Data Summary:")
        st.dataframe(df.describe(), use_container_width=True)

    st.write("Scatter Plot of the raw data:")
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.scatter(df['X'], df['y'], alpha=0.6, edgecolors='k')
    ax.set_xlabel('X')
    ax.set_ylabel('y')
    ax.set_title('Target y vs Feature X')
    st.pyplot(fig)

    # --- 3. Data Preparation ---
    st.header("3. Data Preparation")
    st.write("We will split the data into training (80%) and testing (20%) sets, and apply standard scaling to the feature.")
    
    X_array = df[['X']].values
    y_array = df['y'].values
    
    X_train, X_test, y_train, y_test = train_test_split(X_array, y_array, test_size=0.2, random_state=params['seed'])
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    st.write(f"Training set size: {X_train.shape[0]} samples")
    st.write(f"Testing set size: {X_test.shape[0]} samples")

    # --- 4. Modeling ---
    st.header("4. Modeling")
    st.write("We are fitting a standard Ordinary Least Squares (OLS) Linear Regression model to our scaled training data.")
    
    model = LinearRegression()
    model.fit(X_train_scaled, y_train)
    
    a_learned_scaled = model.coef_[0]
    b_learned_scaled = model.intercept_
    
    # To get unscaled coefficients for comparison:
    a_learned_unscaled = a_learned_scaled / scaler.scale_[0]
    b_learned_unscaled = b_learned_scaled - (a_learned_scaled * scaler.mean_[0] / scaler.scale_[0])
    
    col3, col4 = st.columns(2)
    with col3:
        st.success("**True Parameters**")
        st.write(f"Slope (a): `{a_true:.4f}`")
        st.write(f"Intercept (b): `{b_true:.4f}`")
    with col4:
        st.info("**Learned Parameters (Unscaled equivalent)**")
        st.write(f"Slope (a): `{a_learned_unscaled:.4f}`")
        st.write(f"Intercept (b): `{b_learned_unscaled:.4f}`")

    # --- 5. Evaluation ---
    st.header("5. Evaluation")
    
    y_pred_train = model.predict(X_train_scaled)
    y_pred_test = model.predict(X_test_scaled)
    
    mse = mean_squared_error(y_test, y_pred_test)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred_test)
    
    st.write("Performance Metrics on Test Data:")
    metric_col1, metric_col2, metric_col3 = st.columns(3)
    metric_col1.metric("MSE", f"{mse:.4f}")
    metric_col2.metric("RMSE", f"{rmse:.4f}")
    metric_col3.metric("R² Score", f"{r2:.4f}")
    
    st.write("Regression Line Fit (Test Data):")
    fig2, ax2 = plt.subplots(figsize=(8, 4))
    ax2.scatter(X_test, y_test, color='blue', alpha=0.5, label='Actual Test Data')
    ax2.plot(X_test, y_pred_test, color='red', linewidth=2, label='Regression Line')
    ax2.set_xlabel('X')
    ax2.set_ylabel('y')
    ax2.set_title('Test Data vs Model Prediction')
    ax2.legend()
    st.pyplot(fig2)

    # --- 6. Deployment ---
    st.header("6. Deployment")
    st.write("Our model is now 'deployed' and ready to make predictions on new inputs. We can also save the trained model pipeline.")
    
    # Prediction input
    st.subheader("Make a Prediction")
    user_input = st.number_input("Enter a value for X:", value=0.0, step=1.0)
    user_input_scaled = scaler.transform([[user_input]])
    prediction = model.predict(user_input_scaled)[0]
    st.write(f"Predicted value for y: **{prediction:.4f}**")
    
    # Save model
    st.subheader("Save Model")
    if st.button("Generate Model Artifact (Joblib)"):
        model_artifact = {'model': model, 'scaler': scaler}
        buffer = io.BytesIO()
        joblib.dump(model_artifact, buffer)
        buffer.seek(0)
        
        st.download_button(
            label="Download Model (.joblib)",
            data=buffer,
            file_name="linear_regression_pipeline.joblib",
            mime="application/octet-stream"
        )
else:
    st.info("👈 Please click **Generate Data** in the sidebar to start the CRISP-DM workflow.")
