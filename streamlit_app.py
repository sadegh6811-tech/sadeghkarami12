import streamlit as st
import numpy as np
import sys, os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from discover import discover_equations

st.set_page_config(page_title="MathBot", page_icon="🔬", layout="wide")

st.title("🔬 MathBot")
st.markdown("### Symbolic Discovery of Implicit Manifold Equations")
st.write("Paste your points (one point per line, comma-separated) below to discover the underlying equation.")

# Sidebar
st.sidebar.header("Parameters")
n_eqs = st.sidebar.slider("Number of equations", 1, 5, 1)
degree = st.sidebar.slider("Polynomial degree", 1, 4, 2)
use_trig = st.sidebar.checkbox("Enable trigonometric detection", value=False)

# Text area for data
default_data = "\n".join([f"{np.cos(t):.6f},{np.sin(t):.6f}" for t in np.linspace(0, 2*np.pi, 100)])
data_text = st.text_area("Paste CSV Data (or use default circle)", value=default_data, height=200)

if st.button("🔍 Discover Equation", type="primary"):
    if not data_text.strip():
        st.error("Please provide data.")
    else:
        try:
            # Parse data
            rows = []
            for line in data_text.strip().split("\n"):
                parts = line.replace(";", ",").split(",")
                try:
                    vals = [float(p) for p in parts if p.strip()]
                    if len(vals) >= 2:
                        rows.append(vals)
                except ValueError:
                    continue
            
            X = np.array(rows)
            if len(X) < 50:
                st.error(f"Need at least 50 valid points. Got {len(X)}.")
            else:
                st.info(f"Loaded {len(X)} points, {X.shape[1]} dimensions. Running MathBot...")
                
                # Standardize
                mu = X.mean(axis=0)
                sd = X.std(axis=0)
                sd[sd < 1e-12] = 1.0
                Xs = (X - mu) / sd
                
                # Run MathBot
                res = discover_equations(Xs, n_equations=n_eqs, degree=degree, tol_ratio=0.05, use_trig=use_trig)
                eqs = res[0]
                
                if not eqs:
                    st.warning("No equations discovered.")
                else:
                    st.success(f"Found {len(eqs)} equations!")
                    for i, eq in enumerate(eqs):
                        st.code(str(eq), language="python")
                        
        except Exception as e:
            st.error(f"Error: {e}")

st.markdown("---")
st.markdown("**Code:** [GitHub](https://github.com/sadegh6811-tech/sadeghkarami12)")
