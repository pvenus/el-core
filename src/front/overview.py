

import numpy as np
import pandas as pd
import streamlit as st


def _make_demo_data(seed: int = 42):
    """Create deterministic demo data so the Overview looks real even before wiring the backend."""
    rng = np.random.default_rng(seed)

    n_agents = 120
    n_traits = 6
    n_scenarios = 6

    agents = pd.DataFrame(
        {
            "agent_id": [f"A-{i:03d}" for i in range(1, n_agents + 1)],
        }
    )

    # Traits in [0, 1]
    trait_cols = ["aggression", "fear", "curiosity", "risk", "patience", "cooperation"]
    traits = rng.beta(2.0, 2.0, size=(n_agents, n_traits))
    for j, c in enumerate(trait_cols):
        agents[c] = traits[:, j]

    # Simple clustering (just for visualization)
    agents["cluster"] = pd.cut(agents["risk"] * 0.7 + agents["aggression"] * 0.3, 4, labels=["C1", "C2", "C3", "C4"]).astype(str)

    # Fake 2D embedding (pretend PCA/UMAP)
    x = (agents["risk"] - agents["fear"]) + rng.normal(0, 0.05, size=n_agents)
    y = (agents["aggression"] - agents["patience"]) + rng.normal(0, 0.05, size=n_agents)
    agents["emb_x"] = x
    agents["emb_y"] = y

    # Scenario → action distribution (Attack / Defend / Retreat)
    actions = ["Attack", "Defend", "Retreat"]
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

    return agents, scenarios, actions, trait_cols, explainability


def render_overview():
    st.title("Overview")
    st.caption("Proof in one screen: **agents exist → actions differ → reasons (traits) explain**.")

    # --- Demo data (swap to real data later) ---
    agents, scenarios, actions, trait_cols, explainability = _make_demo_data(
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
        st.subheader("Agent diversity map")
        st.caption("Each dot is an agent. Color shows a simple trait-derived cluster (demo).")

        # Scatter chart using Streamlit native charting (stable, no extra deps)
        chart_df = agents[["emb_x", "emb_y", "cluster", "agent_id"]].copy()
        st.scatter_chart(
            chart_df,
            x="emb_x",
            y="emb_y",
            color="cluster",
        )

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