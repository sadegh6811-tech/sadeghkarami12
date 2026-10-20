import streamlit as st
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    from final_engine_v8_clean import final_engine
    ENGINE_AVAILABLE = True
except Exception:
    ENGINE_AVAILABLE = False

st.set_page_config(page_title="MathBot", page_icon="🔬", layout="wide")

st.title("🔬 MathBot")
st.markdown("### Universal Law Discovery Engine")
st.write("MathBot uses Genetic Programming and a Physics/Math Seed Core to reverse-engineer equations from raw data. Select a dataset below to see it in action.")

# ============================================================
# Sidebar
# ============================================================
st.sidebar.header("Select a Dataset")
dataset = st.sidebar.selectbox(
    "Choose a famous scientific law:",
    (
        "2D Circle (Geometry)",
        "3D Sphere (Geometry)",
        "Newtonian Gravity (Physics)",
        "Gaussian Distribution (Statistics)",
    )
)

st.sidebar.header("Parameters")
n_eqs = st.sidebar.slider("Number of equations to find", 1, 3, 1)
# مقدار درجه و تریگ برای دموی عمومی فیکس می‌شود
degree = 3
use_trig = False

# ============================================================
# Generate Data
# ============================================================
np.random.seed(42)

if dataset == "2D Circle (Geometry)":
    t = np.random.uniform(0, 2*np.pi, 200)
    X = np.column_stack([np.cos(t), np.sin(t)])
    y = np.ones(200) * 1.0
    true_eq = "x_0² + x_1² = 1"

elif dataset == "3D Sphere (Geometry)":
    u = np.random.randn(200, 3)
    u = u / np.linalg.norm(u, axis=1, keepdims=True)
    X = u
    y = np.ones(200) * 1.0
    true_eq = "x_0² + x_1² + x_2² = 1"

elif dataset == "Newtonian Gravity (Physics)":
    m1 = np.random.uniform(0.5, 2.0, 200)
    m2 = np.random.uniform(0.5, 2.0, 200)
    r = np.random.uniform(1.0, 5.0, 200)
    X = np.column_stack([m1, m2, r])
    y = (m1 * m2) / (r**2)
    true_eq = "F = G · (m_1 · m_2) / r²"

elif dataset == "Gaussian Distribution (Statistics)":
    x = np.random.uniform(-3, 3, 200)
    X = x.reshape(-1, 1)
    y = np.exp(-(x**2) / 2)
    true_eq = "y = exp(-x² / 2)"

# ============================================================
# Run MathBot
# ============================================================
st.markdown("---")
st.markdown(f"#### Data: {dataset}")
st.info(f"**True underlying law:** {true_eq}")
st.write(f"Loaded **{len(X)} points** with **{X.shape[1]} dimensions**.")

if st.button("🔍 Discover Equation", type="primary"):
    if not ENGINE_AVAILABLE:
        st.error("Engine not found. Please ensure final_engine_v8_clean.py is present.")
    else:
        with st.spinner("MathBot is analyzing the data..."):
            try:
                result = final_engine(X, y)
                
                best_r2 = -np.inf
                best_eq = "No equation found."
                
                if isinstance(result, dict):
                    best_r2 = result.get('r2', -np.inf)
                    best_eq = result.get('equation', '')
                elif isinstance(result, (tuple, list)):
                    for item in result:
                        if isinstance(item, (int, float)) and -1.5 < item <= 1.0:
                            best_r2 = item
                            break
                    for item in result:
                        if isinstance(item, str) and ("x" in item or "+" in item):
                            best_eq = item
                            break
                elif isinstance(result, (int, float)):
                    best_r2 = result
                    best_eq = "(Equation not returned as string)"
                
                st.success(f"✅ Equation Discovered!")
                st.markdown(f"**Discovered Equation:** `{best_eq}`")
                st.markdown(f"**R² Score:** `{best_r2:.4f}`")
                
                if best_r2 > 0.95:
                    st.balloons()
                    
            except Exception as e:
                st.error(f"Error during discovery: {e}")

st.markdown("---")
st.markdown("**Code:** [GitHub Repository](https://github.com/sadegh6811-tech/sadeghkarami12)")
st.markdown("**Paper:** MathBot v3.1 (JOSS pending)")
