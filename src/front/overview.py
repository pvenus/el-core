import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from sklearn.decomposition import PCA
try:
    import umap
except Exception:  # pragma: no cover
    umap = None


def _make_demo_data(seed: int = 42):
    """Create deterministic demo data so the Overview looks real even before wiring the backend."""
    rng = np.random.default_rng(seed)

    n_agents = 120
    n_traits = 27

    agents = pd.DataFrame(
        {
            "agent_id": [f"A-{i:03d}" for i in range(1, n_agents + 1)],
        }
    )

    # 27D vector in [-1, 1] (demo).
    # Name axes as a..z plus aa (27 dims) then group into 9 triplets (3D subspaces).
    axis_names = [
        "a","b","c","d","e","f","g","h","i","j","k","l","m","n","o",
        "p","q","r","s","t","u","v","w","x","y","z","aa"
    ]
    trait_cols = axis_names

    # Generate values in [-1, 1]
    traits = rng.uniform(-1.0, 1.0, size=(n_agents, n_traits))
    for j, c in enumerate(trait_cols):
        agents[c] = traits[:, j]

    # 9 subspaces: (a,b,c), (d,e,f), ...
    subspaces = [
        ("a","b","c"),
        ("d","e","f"),
        ("g","h","i"),
        ("j","k","l"),
        ("m","n","o"),
        ("p","q","r"),
        ("s","t","u"),
        ("v","w","x"),
        ("y","z","aa"),
    ]

    agents["cluster"] = pd.cut(agents["e"] * 0.5 + agents["t"] * 0.3 - agents["b"] * 0.2, 4, labels=["C1", "C2", "C3", "C4"]).astype(str)

    # Scenario → action distribution (Attack / Defend / Retreat)
    actions = ["Attack", "Defend", "Retreat"]
    n_scenarios = 6
    scenarios = []
    for s in range(1, n_scenarios + 1):
        # each scenario has a different base distribution
        base = rng.dirichlet([2.5, 2.0, 1.8])
        # compute a crude "diversity" score for the scenario
        diversity = float(-(base * np.log(base + 1e-9)).sum())
        scenarios.append(
            {
                "scenario_id": f"S-{s:02d}",
                "name": [
                    "Low HP / High Threat",
                    "Even Fight",
                    "Outnumbered",
                    "Ally Down",
                    "Enemy Weak",
                    "Resource Starved",
                ][s - 1],
                "p_attack": float(base[0]),
                "p_defend": float(base[1]),
                "p_retreat": float(base[2]),
                "diversity": diversity,
            }
        )

    scenarios = pd.DataFrame(scenarios)

    # Pretend we measured explainability as agreement between trait-rule and choice
    explainability = float(rng.uniform(0.62, 0.83))

    return agents, scenarios, actions, trait_cols, subspaces, explainability


def render_overview():
    st.title("Overview")
    st.caption("Proof in one screen: **agents exist → actions differ → reasons (traits) explain**.")

    # --- Demo data (swap to real data later) ---
    agents, scenarios, actions, trait_cols, subspaces, explainability = _make_demo_data(
        seed=st.session_state.get("demo_seed", 42)
    )

    # --- Top KPIs ---
    st.subheader("At a glance")
    c1, c2, c3, c4 = st.columns(4)

    # Average unique actions per scenario (demo)
    avg_unique = float((scenarios[["p_attack", "p_defend", "p_retreat"]] > 0.15).sum(axis=1).mean())

    c1.metric("Agents", f"{len(agents):,}")
    c2.metric("Scenarios", f"{len(scenarios):,}")
    c3.metric("Avg unique actions / scenario", f"{avg_unique:.1f}")
    c4.metric("Explainability (trait→choice)", f"{explainability:.2f}")

    st.divider()

    # --- Middle: diversity map + short narrative ---
    left, right = st.columns([1.35, 1.0])

    with left:
        st.subheader("Agent vector spaces")
        st.caption(
            "Each agent is a 27D vector in [-1, 1]. We show it as **9 separate 3D subspaces** to keep the geometry interpretable."
        )
        st.divider()
        st.subheader("27D split into 9 × 3D subspaces")
        st.caption(
            "All 9 subspaces are rendered in **true 3D** at once (3×3 grid). "
            "Pick an agent to highlight its position and direction vector in every subspace."
        )

        # Agent selector for highlighting across all 9 3D panels
        agent_id_3d = st.selectbox(
            "Highlight agent (applies to all panels)",
            agents["agent_id"].tolist(),
            index=0,
            key="overview_agent_pick_3d",
        )
        arow3d = agents.loc[agents["agent_id"] == agent_id_3d].iloc[0]

        show_vectors = st.checkbox("Show direction vectors (origin → agent)", value=True, key="overview_show_vectors")

        # Render 9 true-3D plots in a 3×3 grid
        grid_rows = [st.columns(3) for _ in range(3)]
        clusters = sorted(agents["cluster"].unique())

        for idx, (xcol, ycol, zcol) in enumerate(subspaces):
            r = idx // 3
            c = idx % 3
            with grid_rows[r][c]:
                fig3d = go.Figure()

                # Per-cluster point clouds
                for cl in clusters:
                    dfc = agents.loc[agents["cluster"] == cl, ["agent_id", xcol, ycol, zcol]].copy()
                    fig3d.add_trace(
                        go.Scatter3d(
                            x=dfc[xcol],
                            y=dfc[ycol],
                            z=dfc[zcol],
                            mode="markers",
                            name=str(cl),
                            marker=dict(size=2.8, opacity=0.70),
                            text=dfc["agent_id"],
                            hovertemplate=(
                                "<b>%{text}</b><br>"
                                + f"{xcol}=%{{x:.3f}}<br>{ycol}=%{{y:.3f}}<br>{zcol}=%{{z:.3f}}<extra></extra>"
                            ),
                        )
                    )

                # Highlight selected agent
                fig3d.add_trace(
                    go.Scatter3d(
                        x=[float(arow3d[xcol])],
                        y=[float(arow3d[ycol])],
                        z=[float(arow3d[zcol])],
                        mode="markers",
                        name=f"Selected",
                        marker=dict(size=7, symbol="diamond", opacity=1.0),
                        hovertemplate=(
                            f"<b>{agent_id_3d}</b><br>"
                            + f"{xcol}=%{{x:.3f}}<br>{ycol}=%{{y:.3f}}<br>{zcol}=%{{z:.3f}}<extra></extra>"
                        ),
                        showlegend=False,
                    )
                )

                # Optional: direction vector from origin
                if show_vectors:
                    fig3d.add_trace(
                        go.Scatter3d(
                            x=[0.0, float(arow3d[xcol])],
                            y=[0.0, float(arow3d[ycol])],
                            z=[0.0, float(arow3d[zcol])],
                            mode="lines",
                            name="Direction",
                            line=dict(width=5),
                            hoverinfo="skip",
                            showlegend=False,
                        )
                    )

                fig3d.update_layout(
                    margin=dict(l=0, r=0, t=28, b=0),
                    height=260,
                    showlegend=False,
                    title=f"S{idx+1}: ({xcol},{ycol},{zcol})",
                    scene=dict(
                        xaxis=dict(range=[-1.05, 1.05], title=xcol, zeroline=True, showticklabels=False),
                        yaxis=dict(range=[-1.05, 1.05], title=ycol, zeroline=True, showticklabels=False),
                        zaxis=dict(range=[-1.05, 1.05], title=zcol, zeroline=True, showticklabels=False),
                        aspectmode="cube",
                    ),
                )

                st.plotly_chart(fig3d, use_container_width=True)

        st.divider()
        st.subheader("Inspect an agent vector")
        agent_id = st.selectbox("Agent", agents["agent_id"].tolist(), index=0, key="overview_agent_pick")
        arow = agents.loc[agents["agent_id"] == agent_id].iloc[0]

        # Compute per-subspace magnitude (L2 norm) to show how the agent "leans" across spaces
        mags = []
        labels = []
        for i, (xcol, ycol, zcol) in enumerate(subspaces, start=1):
            v = np.array([arow[xcol], arow[ycol], arow[zcol]], dtype=float)
            mags.append(float(np.linalg.norm(v)))
            labels.append(f"S{i}({xcol}{ycol}{zcol})")

        mag_df = pd.DataFrame({"subspace": labels, "magnitude": mags})
        st.bar_chart(mag_df, x="subspace", y="magnitude", height=240)

        with st.expander("Show selected agent raw 27D vector", expanded=False):
            st.write({k: float(arow[k]) for k in trait_cols})

        with st.expander("Show agent table (demo)", expanded=False):
            cols = ["agent_id", "cluster"] + trait_cols
            st.dataframe(agents[cols].head(30), use_container_width=True)

    with right:
        st.subheader("What this proves")
        st.markdown(
            """
- **Different agents are generated** (trait vectors differ).
- Under the **same scenario**, agents show **different action distributions**.
- Differences are **explainable** via trait signals (e.g., risk↑ → attack bias, fear↑ → retreat bias).
- The same agent has a **27D signature**, shown as **9 separate 3D subspaces** for interpretability.

**Goal for IR / evaluation:** show *reproducible* evidence with logs + measurable metrics.
            """
        )

        st.divider()
        st.subheader("Quick demo controls")
        st.caption("These controls only change demo data. Wire them to real runs later.")
        st.slider("Demo seed", 1, 999, int(st.session_state.get("demo_seed", 42)), key="demo_seed")
        st.checkbox("Normalize trait values (0..1)", value=True, disabled=True)

    st.divider()

    # --- Bottom: top scenarios cards ---
    st.subheader("Representative scenarios")
    st.caption("Pick a scenario: see action distribution + which traits typically drive divergence.")

    # Choose top 3 by diversity
    top = scenarios.sort_values("diversity", ascending=False).head(3).reset_index(drop=True)

    cols = st.columns(3)
    for i in range(3):
        row = top.iloc[i]
        with cols[i]:
            st.markdown(f"### {row['scenario_id']}  ")
            st.caption(row["name"])

            # Action distribution (as a small bar chart)
            adf = pd.DataFrame(
                {
                    "action": actions,
                    "prob": [row["p_attack"], row["p_defend"], row["p_retreat"]],
                }
            )
            st.bar_chart(adf, x="action", y="prob", height=160)

            # Heuristic "driver traits" (demo explanation)
            if "Low HP" in row["name"] or "Outnumbered" in row["name"] or "Ally" in row["name"]:
                drivers = ["fear", "patience", "cooperation"]
            elif "Enemy Weak" in row["name"]:
                drivers = ["aggression", "risk", "curiosity"]
            else:
                drivers = ["risk", "aggression", "fear"]

            st.markdown("**Likely driver traits**")
            st.write(", ".join([f"`{d}`" for d in drivers]))

    st.divider()

    # --- Export section (for business plan / screenshots) ---
    st.subheader("Export")
    st.caption("Use these to paste evidence into the business plan / IR deck.")

    exp1, exp2 = st.columns([1, 1])
    with exp1:
        st.download_button(
            "Download demo agents.csv",
            data=agents.to_csv(index=False).encode("utf-8"),
            file_name="agents_demo.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with exp2:
        st.download_button(
            "Download demo scenarios.csv",
            data=scenarios.to_csv(index=False).encode("utf-8"),
            file_name="scenarios_demo.csv",
            mime="text/csv",
            use_container_width=True,
        )

    st.info(
        "Next: connect real logs (runs.parquet/csv) and replace the demo generator with your pipeline outputs."
    )