import os
import sys

# Ensure local imports work when running: streamlit run src/front/main.py
sys.path.append(os.path.dirname(__file__))

import streamlit as st
from overview import render_overview

# -------------------------
# Streamlit App: EL-Core UI
# -------------------------

st.set_page_config(
    page_title="EL-Core Dashboard",
    page_icon="🧠",
    layout="wide",
)

# Sidebar navigation (button-based)
st.sidebar.title("EL-Core")

if "menu" not in st.session_state:
    st.session_state.menu = "Overview"


def nav_button(label):
    is_active = st.session_state.menu == label
    # Use button styling instead of injecting extra markdown (which shifts layout on reruns)
    clicked = st.sidebar.button(
        label,
        use_container_width=True,
        type="primary" if is_active else "secondary",
        key=f"nav_{label}",
    )
    if clicked:
        st.session_state.menu = label
        st.rerun()

nav_button("Overview")
nav_button("Agent Gallery")
nav_button("Scenario Lab")
nav_button("Compare")
nav_button("Analytics")

menu = st.session_state.menu

# Optional: global controls (placeholders)
st.sidebar.divider()
with st.sidebar.expander("Global Filters", expanded=False):
    st.selectbox("Dataset", ["demo"], index=0)
    st.checkbox("Use cached data", value=True)


def render_agent_gallery():
    st.title("Agent Gallery")
    st.caption("Browse agents and their traits / behavior fingerprints.")
    st.info("TODO: Add filters + agent cards + detail panel.")


def render_scenario_lab():
    st.title("Scenario Lab")
    st.caption("Hold state constant, swap agents, observe diverging actions.")
    st.info("TODO: Add scenario selector + state editor + action heatmap.")


def render_compare():
    st.title("Compare")
    st.caption("Counterfactual matrix: similar reasoning → different action, and vice versa.")
    st.info("TODO: Add 2x2 matrix + A/B agent comparison panel.")


def render_analytics():
    st.title("Analytics")
    st.caption("Quantitative evidence: diversity, correlations, clusters, learning curves.")
    st.info("TODO: Add charts + export buttons.")


# Route
if menu == "Overview":
    render_overview()
elif menu == "Agent Gallery":
    render_agent_gallery()
elif menu == "Scenario Lab":
    render_scenario_lab()
elif menu == "Compare":
    render_compare()
elif menu == "Analytics":
    render_analytics()
