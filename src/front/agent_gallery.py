from typing import Optional
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go


def _get_demo_population():
    """Reuse Overview demo data if available; otherwise generate a small deterministic demo set."""
    try:
        from overview import _make_demo_data  # type: ignore

        agents, _scenarios, _actions, trait_cols, subspaces, _explain = _make_demo_data(
            seed=st.session_state.get("demo_seed", 42)
        )
        # Force cluster to align with the 9×3D split (S1..S9)
        mags = []
        for (xcol, ycol, zcol) in subspaces:
            v = agents[[xcol, ycol, zcol]].to_numpy(dtype=float)
            mags.append(np.linalg.norm(v, axis=1))
        mags = np.stack(mags, axis=1)

        agents["dominant_subspace"] = mags.argmax(axis=1) + 1
        agents["dominant_mag"] = mags.max(axis=1)
        agents["cluster"] = agents["dominant_subspace"].apply(lambda i: f"S{i}")

        return agents, trait_cols, subspaces
    except Exception:
        rng = np.random.default_rng(int(st.session_state.get("demo_seed", 42)))
        n_agents = 60
        axis_names = [
            "a","b","c","d","e","f","g","h","i","j","k","l","m","n","o",
            "p","q","r","s","t","u","v","w","x","y","z","aa"
        ]
        trait_cols = axis_names
        agents = pd.DataFrame({"agent_id": [f"A-{i:03d}" for i in range(1, n_agents + 1)]})
        for c in trait_cols:
            agents[c] = rng.uniform(-1.0, 1.0, size=n_agents)
        subspaces = [
            ("a","b","c"), ("d","e","f"), ("g","h","i"),
            ("j","k","l"), ("m","n","o"), ("p","q","r"),
            ("s","t","u"), ("v","w","x"), ("y","z","aa"),
        ]
        # Cluster aligned to the 9×3D split: dominant subspace by L2 magnitude
        mags = []
        for (xcol, ycol, zcol) in subspaces:
            v = agents[[xcol, ycol, zcol]].to_numpy(dtype=float)
            mags.append(np.linalg.norm(v, axis=1))
        mags = np.stack(mags, axis=1)

        agents["dominant_subspace"] = mags.argmax(axis=1) + 1
        agents["dominant_mag"] = mags.max(axis=1)
        agents["cluster"] = agents["dominant_subspace"].apply(lambda i: f"S{i}")
        return agents, trait_cols, subspaces


def _subspace_magnitudes(row: pd.Series, subspaces):
    mags = []
    for (x, y, z) in subspaces:
        v = np.array([float(row[x]), float(row[y]), float(row[z])], dtype=float)
        mags.append(float(np.linalg.norm(v)))
    return mags


def _render_subspace_grid(mags, dominant_idx: int):
    """Render a compact 3×3 grid for S1..S9 magnitudes.

    `mags` must be a length-9 list[float].
    `dominant_idx` is 1-based (S1..S9).
    """
    if not mags:
        return

    mmax = max(mags)
    mmax = mmax if mmax > 0 else 1.0

    def cell_html(i: int, v: float) -> str:
        # Normalize for shading
        norm = min(max(v / mmax, 0.0), 1.0)
        shade = int(240 - 140 * norm)  # 240 (light) -> 100 (darker)
        border = "2px solid #111" if i == dominant_idx else "1px solid #ddd"
        return (
            f"<div style='border:{border};border-radius:10px;padding:6px 8px;"
            f"background:rgb({shade},{shade},{shade});'>"
            f"<div style='font-size:11px;opacity:0.85'>S{i}</div>"
            f"<div style='font-size:13px;font-weight:700;line-height:1.1'>{v:.2f}</div>"
            f"</div>"
        )

    cells = [cell_html(i + 1, float(mags[i])) for i in range(min(9, len(mags)))]
    # Pad to 9 cells if needed
    while len(cells) < 9:
        cells.append(cell_html(len(cells) + 1, 0.0))

    html = (
        "<div style='display:grid;grid-template-columns:repeat(3,1fr);gap:6px;'>"
        + "".join(cells[:9])
        + "</div>"
    )
    st.markdown(html, unsafe_allow_html=True)


# --- 3x3 grid of 3D subspace scatter plots ---
def _render_subspace_3x3(df_plot: pd.DataFrame, subspaces, selected_agent_id: Optional[str], show_vectors: bool = True):
    """Render 9 subspaces as a 3×3 grid of 3D scatter plots."""
    grid_rows = [st.columns(3) for _ in range(3)]

    # Selected row (may be None)
    sel_row = None
    if selected_agent_id is not None and len(df_plot) > 0:
        hit = df_plot.loc[df_plot["agent_id"] == selected_agent_id]
        if len(hit) > 0:
            sel_row = hit.iloc[0]

    # Ensure clusters exist for legend grouping
    clusters = sorted(df_plot["cluster"].unique()) if (len(df_plot) > 0 and "cluster" in df_plot.columns) else []

    for idx, (xcol, ycol, zcol) in enumerate(subspaces):
        r = idx // 3
        c = idx % 3
        with grid_rows[r][c]:
            fig3d = go.Figure()

            # Plot by cluster (dominant subspace) for readability
            if len(clusters) > 0:
                for cl in clusters:
                    dfc = df_plot.loc[df_plot["cluster"] == cl, ["agent_id", xcol, ycol, zcol]].copy()
                    fig3d.add_trace(
                        go.Scatter3d(
                            x=dfc[xcol],
                            y=dfc[ycol],
                            z=dfc[zcol],
                            mode="markers",
                            name=str(cl),
                            marker=dict(size=2.6, opacity=0.70),
                            text=dfc["agent_id"],
                            hovertemplate=(
                                "<b>%{text}</b><br>"
                                + f"{xcol}=%{{x:.3f}}<br>{ycol}=%{{y:.3f}}<br>{zcol}=%{{z:.3f}}<extra></extra>"
                            ),
                            showlegend=False,
                        )
                    )
            else:
                fig3d.add_trace(
                    go.Scatter3d(
                        x=df_plot[xcol],
                        y=df_plot[ycol],
                        z=df_plot[zcol],
                        mode="markers",
                        marker=dict(size=2.6, opacity=0.70),
                        text=df_plot["agent_id"],
                        hovertemplate=(
                            "<b>%{text}</b><br>"
                            + f"{xcol}=%{{x:.3f}}<br>{ycol}=%{{y:.3f}}<br>{zcol}=%{{z:.3f}}<extra></extra>"
                        ),
                        showlegend=False,
                    )
                )

            # Selected agent highlight
            if sel_row is not None:
                fig3d.add_trace(
                    go.Scatter3d(
                        x=[float(sel_row[xcol])],
                        y=[float(sel_row[ycol])],
                        z=[float(sel_row[zcol])],
                        mode="markers",
                        marker=dict(size=7, symbol="diamond", opacity=1.0),
                        hovertemplate=(
                            f"<b>{selected_agent_id}</b><br>"
                            + f"{xcol}=%{{x:.3f}}<br>{ycol}=%{{y:.3f}}<br>{zcol}=%{{z:.3f}}<extra></extra>"
                        ),
                        showlegend=False,
                    )
                )

                if show_vectors:
                    fig3d.add_trace(
                        go.Scatter3d(
                            x=[0.0, float(sel_row[xcol])],
                            y=[0.0, float(sel_row[ycol])],
                            z=[0.0, float(sel_row[zcol])],
                            mode="lines",
                            line=dict(width=5),
                            hoverinfo="skip",
                            showlegend=False,
                        )
                    )

            fig3d.update_layout(
                margin=dict(l=0, r=0, t=28, b=0),
                height=260,
                title=f"S{idx+1}: ({xcol},{ycol},{zcol})",
                scene=dict(
                    xaxis=dict(range=[-1.05, 1.05], title=xcol, zeroline=True, showticklabels=False),
                    yaxis=dict(range=[-1.05, 1.05], title=ycol, zeroline=True, showticklabels=False),
                    zaxis=dict(range=[-1.05, 1.05], title=zcol, zeroline=True, showticklabels=False),
                    aspectmode="cube",
                ),
                showlegend=False,
            )

            st.plotly_chart(fig3d, use_container_width=True)


def render_agent_gallery():
    st.title("Agent Gallery")
    st.caption("Browse agent identities (27D vectors), filter by clusters/traits, and inspect a chosen agent in detail.")

    agents, trait_cols, subspaces = _get_demo_population()

    # --- Layout ---
    filters_col, grid_col, detail_col = st.columns([0.95, 1.6, 1.15])

    # --- Filters ---
    with filters_col:
        st.subheader("Explore")

        # Build subspace label -> triplet mapping (S1..S9)
        subspace_labels = [f"S{i+1}" for i in range(len(subspaces))]
        subspace_triplets = {f"S{i+1}": subspaces[i] for i in range(len(subspaces))}

        explore_mode = st.radio(
            "Mode",
            ["Subspace-first (recommended)", "Axis-first"],
            index=0,
            help=(
                "Subspace-first: pick one 3D subspace (S1..S9), then inspect its 3 axes together. "
                "Axis-first: pick any single axis across the full 27D and slice by range."
            ),
        )

        # --- Subspace-first ---
        if explore_mode.startswith("Subspace-first"):
            chosen_subspace = st.selectbox(
                "Subspace (3D)",
                ["All"] + subspace_labels,
                index=0,
                help="Each S# is one 3D block (e.g., S3 = (g,h,i)). Choose All to see every agent.",
            )

            # When showing All, default the axis triplet to S1 for coherent axis selection UI.
            axis_subspace_for_ui = "S1" if chosen_subspace == "All" else chosen_subspace
            ax_triplet = subspace_triplets[axis_subspace_for_ui]
            axis = st.selectbox(
                "Axis in this subspace",
                list(ax_triplet),
                index=0,
                help="Filter uses only the selected axis inside the chosen 3D block.",
            )

            lo, hi = st.slider(
                "Axis range",
                -1.0,
                1.0,
                (-1.0, 1.0),
                step=0.05,
            )

            # Sorting
            sort_by = st.selectbox(
                "Sort by",
                ["agent_id", "dominant_mag", f"axis:{axis}"],
                index=0,
            )
            ascending = st.checkbox("Ascending", value=True)

            st.divider()
            st.caption(
                "Interpretation: you are viewing agents through one coherent 3D lens. "
                "Cluster(S#) means the agent's *dominant* subspace by L2 magnitude."
            )

        # --- Axis-first ---
        else:
            axis = st.selectbox(
                "Axis (27D)",
                trait_cols,
                index=trait_cols.index("e") if "e" in trait_cols else 0,
                help="Choose any axis across the full 27D vector.",
            )
            lo, hi = st.slider("Axis range", -1.0, 1.0, (-1.0, 1.0), step=0.05)

            # Optional: keep cluster filter, but now it has a clear meaning (dominant subspace)
            clusters = sorted(agents["cluster"].unique())
            sel_clusters = st.multiselect(
                "Dominant subspace (optional)",
                clusters,
                default=clusters,
                help="Filter by the agent's dominant subspace (S1..S9).",
            )

            sort_by = st.selectbox(
                "Sort by",
                ["agent_id", "cluster", f"axis:{axis}", "dominant_mag"],
                index=0,
            )
            ascending = st.checkbox("Ascending", value=True)

            st.divider()
            st.caption(
                "Use Axis-first when you want to ask a single trait question (e.g., 'g is high'). "
                "Use Dominant subspace filter to narrow to a coherent personality family."
            )

    # Apply filters
    df = agents.copy()

    # Axis range filter (always)
    df = df[(df[axis] >= lo) & (df[axis] <= hi)]

    # Mode-specific filters
    if "explore_mode" in locals() and explore_mode.startswith("Subspace-first"):
        # In subspace-first mode, you may either narrow to one dominant subspace (S#) or show All.
        if chosen_subspace != "All":
            df = df[df["cluster"] == chosen_subspace]
    else:
        # In axis-first mode, optionally filter by dominant subspace list.
        df = df[df["cluster"].isin(sel_clusters)]

    # Diagnostics: make filter effects explicit
    total_n = len(agents)
    after_axis_n = len(agents[(agents[axis] >= lo) & (agents[axis] <= hi)])
    after_all_filters_n = len(df)

    with filters_col:
        st.markdown("**Filter summary**")
        st.write(
            {
                "total_agents": int(total_n),
                "after_axis_range": int(after_axis_n),
                "after_all_filters": int(after_all_filters_n),
                "mode": explore_mode,
            }
        )

    # Compute dominant subspace magnitude (for sorting / badges)
    mags = df.apply(lambda r: _subspace_magnitudes(r, subspaces), axis=1)
    df = df.assign(
        dominant_subspace=mags.apply(lambda m: int(np.argmax(m)) + 1),
        dominant_mag=mags.apply(lambda m: float(np.max(m))),
    )

    if sort_by == "agent_id":
        df = df.sort_values("agent_id", ascending=ascending)
    elif sort_by == "cluster":
        df = df.sort_values(["cluster", "agent_id"], ascending=ascending)
    elif sort_by == "dominant_mag":
        df = df.sort_values(["dominant_mag", "agent_id"], ascending=ascending)
    elif sort_by.startswith("axis:"):
        df = df.sort_values(axis, ascending=ascending)
    else:
        # Fallback: dominant subspace then magnitude
        df = df.sort_values(["dominant_subspace", "dominant_mag"], ascending=ascending)

    # --- Agents view (27D split into 9×3D subspaces) ---
    with grid_col:
        st.subheader(f"Agents ({len(df):,})")

        if len(df) == 0:
            st.warning("No agents match the current filters.")
        else:
            # Highlight selection for the 3D panels
            highlight_id = st.selectbox(
                "Highlight agent (applies to all 9 panels)",
                df["agent_id"].tolist(),
                index=0,
                key="gallery_highlight_agent",
            )
            st.session_state["selected_agent_id"] = highlight_id
            show_vectors = st.checkbox("Show direction vectors (origin → agent)", value=True, key="gallery_show_vectors")

            _render_subspace_3x3(df, subspaces, highlight_id, show_vectors=show_vectors)

    # --- Detail (single selected agent) ---
    with detail_col:
        st.subheader("Agent details")
        st.caption("Shows the currently selected agent from the middle view (or the last Inspect click).")

        if len(df) == 0:
            st.info("No agent to show (filters removed all).")
            return

        selected_id = st.session_state.get("selected_agent_id")
        if selected_id is None:
            selected_id = df.iloc[0]["agent_id"]
            st.session_state["selected_agent_id"] = selected_id

        hit = df.loc[df["agent_id"] == selected_id]
        if len(hit) == 0:
            # If the selected agent is no longer in the filtered set, fall back to the first row
            hit = df.iloc[[0]]
            selected_id = hit.iloc[0]["agent_id"]
            st.session_state["selected_agent_id"] = selected_id

        r = hit.iloc[0]

        st.markdown(f"### {r['agent_id']}")
        st.caption(f"Cluster: `{r['cluster']}` | Dominant: `S{int(r['dominant_subspace'])}`")

        # Subspace grid signature
        card_mags = _subspace_magnitudes(r, subspaces)
        _render_subspace_grid(card_mags, int(r["dominant_subspace"]))

        st.divider()
        st.markdown("**Subspace signature (L2 magnitude)**")
        mag_df = pd.DataFrame(
            {
                "subspace": [f"S{i+1}:({a},{b},{c})" for i, (a, b, c) in enumerate(subspaces)],
                "magnitude": card_mags,
            }
        )
        st.bar_chart(mag_df, x="subspace", y="magnitude", height=220)

        with st.expander("Raw 27D vector (3×3 subspaces)", expanded=False):
            st.caption("Each card is one 3D subspace (S1..S9). Values are the 3 axes and the L2 magnitude.")
            grid_rows = [st.columns(3) for _ in range(3)]
            for sidx, (x, y, z) in enumerate(subspaces):
                rr = sidx // 3
                cc = sidx % 3
                with grid_rows[rr][cc]:
                    v = np.array([float(r[x]), float(r[y]), float(r[z])], dtype=float)
                    mag = float(np.linalg.norm(v))
                    subcard = st.container(border=True)
                    with subcard:
                        st.markdown(f"**S{sidx+1}**")
                        st.caption(f"({x}, {y}, {z}) | mag={mag:.3f}")
                        st.write({x: float(r[x]), y: float(r[y]), z: float(r[z])})

        st.markdown("**Export**")
        st.download_button(
            "Download this agent as JSON",
            data=pd.Series(
                {"agent_id": r["agent_id"], "cluster": r["cluster"], **{c: float(r[c]) for c in trait_cols}}
            )
            .to_json()
            .encode("utf-8"),
            file_name=f"{r['agent_id']}.json",
            mime="application/json",
            width='stretch',
        )