"""Streamlit browser for completed runs — supports multi-run side-by-side comparison.

Launch with:  streamlit run src/activation_lab/view/app.py
Or pass a default run: streamlit run src/activation_lab/view/app.py -- <run_dir>
Or set ACTIVATION_LAB_RUNS_DIR to override the runs search directory.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

_HERE = Path(__file__).resolve()
_SRC = _HERE.parents[2]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from activation_lab.view import Run  # noqa: E402

# --------------------------------------------------------------------------- helpers

_RUNS_DIR = Path(os.environ.get("ACTIVATION_LAB_RUNS_DIR", "runs"))


@st.cache_resource(show_spinner="Loading run metadata…")
def _load_run(path: str) -> Run:
    return Run(path)


def _discover_runs() -> list[Path]:
    if not _RUNS_DIR.exists():
        return []
    return sorted(
        [p for p in _RUNS_DIR.iterdir() if p.is_dir() and (p / "run.json").exists()],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )


def _seed_from_argv() -> str | None:
    """If a run dir was passed as a CLI arg, return it (legacy compat)."""
    if len(sys.argv) > 1 and not sys.argv[1].startswith("-"):
        p = Path(sys.argv[1])
        if p.exists():
            return str(p)
    env = os.environ.get("ACTIVATION_LAB_RUN")
    if env:
        return env
    return None


# --------------------------------------------------------------------------- per-run renderers

def _render_heatmap(run: Run, step: int, layer: int, source: str, cmap: str) -> None:
    try:
        data = run.tensor(step, layer, source)
        if data.ndim == 3:
            data = data[0]
        elif data.ndim == 4:
            data = data[0].mean(axis=0)
            st.caption("attn_weights shown as head-mean")
    except KeyError:
        st.warning(f"`{source}` not captured for layer {layer}.")
        return
    fig, ax = plt.subplots(figsize=(8, max(2.0, min(5.0, data.shape[0] / 10 + 1.5))))
    vmax = float(np.abs(data).max()) or 1.0
    im = ax.imshow(data, aspect="auto", cmap=cmap, vmin=-vmax, vmax=vmax, interpolation="nearest")
    ax.set_xlabel("channel / key pos")
    ax.set_ylabel("pos / query pos")
    fig.colorbar(im, ax=ax, shrink=0.8)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)
    ca, cb, cc = st.columns(3)
    ca.metric("shape", f"{tuple(data.shape)}")
    cb.metric("min", f"{data.min():.3f}")
    cc.metric("max", f"{data.max():.3f}")


def _render_residual(run: Run, step: int) -> None:
    rows = run.residual_norms(step)
    if not rows:
        st.info("Requires hidden_in + attn_out + mlp_down_out + hidden_out captured.")
        return
    xs = [r["layer"] for r in rows]
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(xs, [r["norm_hidden_in"] for r in rows], label="||h_in||")
    ax.plot(xs, [r["norm_attn_out"] for r in rows], label="||attn||")
    ax.plot(xs, [r["norm_mlp_down_out"] for r in rows], label="||mlp||")
    ax.plot(xs, [r["norm_hidden_out"] for r in rows], label="||h_out||")
    ax.set_xlabel("layer")
    ax.set_ylabel("L2 norm")
    ax.legend()
    ax.grid(True, alpha=0.3)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    fig2, ax2 = plt.subplots(figsize=(8, 2.5))
    ax2.plot(xs, [r["attn_contribution_cos"] for r in rows], label="cos(attn, h_out)")
    ax2.plot(xs, [r["mlp_contribution_cos"] for r in rows], label="cos(mlp, h_out)")
    ax2.axhline(0, color="black", lw=0.5)
    ax2.set_ylim(-1, 1)
    ax2.set_xlabel("layer")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    st.pyplot(fig2, use_container_width=True)
    plt.close(fig2)


def _render_logit_lens(run: Run, step: int, position: int, top_k_n: int, btn_key: str) -> None:
    if st.button("Run logit lens", type="primary", key=btn_key):
        with st.spinner("Projecting hidden states through final_norm + lm_head…"):
            rows = run.logit_lens_table(step, position=position, top_k=top_k_n)
        if rows:
            st.dataframe(
                [
                    {
                        "layer": r["layer"],
                        "argmax": r["argmax_token"],
                        "argmax_logprob": r["argmax_logprob"],
                        "kl_from_final": r["kl_from_final"],
                        "top_k": " ".join(f"{t}({lp})" for t, lp in r["top"]),
                    }
                    for r in rows
                ],
                use_container_width=True,
                hide_index=True,
            )
            kls = [(r["layer"], r["kl_from_final"]) for r in rows if r["kl_from_final"] is not None]
            if kls:
                figl, axl = plt.subplots(figsize=(8, 3))
                axl.plot([k[0] for k in kls], [k[1] for k in kls], marker="o")
                axl.set_xlabel("layer")
                axl.set_ylabel("KL(layer || final)")
                axl.set_title("Prediction convergence across layers")
                axl.grid(True, alpha=0.3)
                st.pyplot(figl, use_container_width=True)
                plt.close(figl)


def _render_neuron(run: Run, step: int, layer: int, n_source: str, neuron: int) -> None:
    try:
        probe = run.tensor(step, layer, n_source)
    except KeyError:
        st.warning(f"`{n_source}` not captured for layer {layer}.")
        return
    D = probe.shape[-1]
    if neuron >= D:
        st.warning(f"Neuron {neuron} out of range (max {D - 1} for this run).")
        return
    vals = run.neuron_trajectory(layer, neuron, source=n_source)
    figN, axN = plt.subplots(figsize=(8, 3))
    xs = np.arange(len(vals))
    axN.plot(xs, vals, marker="o")
    axN.axhline(0, color="black", lw=0.5)
    axN.set_xlabel("step")
    axN.set_ylabel("activation (last pos)")
    axN.set_title(f"L{layer}#{neuron} ({n_source})")
    axN.grid(True, alpha=0.3)
    for i, s in enumerate(run.steps):
        tok = s.generated_token or ""
        axN.annotate(
            tok, (i, vals[i] if not np.isnan(vals[i]) else 0),
            fontsize=7, alpha=0.7, xytext=(0, 4), textcoords="offset points", ha="center",
        )
    st.pyplot(figN, use_container_width=True)
    plt.close(figN)


def _render_cross_layer(run: Run, step: int, cl_source: str) -> None:
    M = run.cross_layer_cosine(step, source=cl_source)
    figC, axC = plt.subplots(figsize=(5, 4))
    im = axC.imshow(M, cmap="RdBu_r", vmin=-1, vmax=1, interpolation="nearest")
    axC.set_xlabel("layer")
    axC.set_ylabel("layer")
    figC.colorbar(im, ax=axC, shrink=0.8)
    st.pyplot(figC, use_container_width=True)
    plt.close(figC)
    st.caption("Bright diagonal = distinct reps. Off-diagonal bands = redundant layers.")


# --------------------------------------------------------------------------- main

def main() -> None:
    st.set_page_config(page_title="Activation Lab", layout="wide")
    st.title("Activation Lab")

    # Seed session state with a CLI-provided run (legacy compat), once.
    if "selected_runs" not in st.session_state:
        st.session_state.selected_runs = []
        seed = _seed_from_argv()
        if seed:
            st.session_state.selected_runs.append(seed)

    # ----------------------------------------------------------------- sidebar: run picker
    st.sidebar.header("Runs")
    available = _discover_runs()
    available_map = {p.name: str(p) for p in available}
    already = set(st.session_state.selected_runs)

    addable = [name for name, path in available_map.items() if path not in already]
    chosen = st.sidebar.selectbox(
        "Add run",
        ["— pick —"] + addable,
        disabled=not addable,
    )
    if st.sidebar.button("Add", disabled=(chosen == "— pick —")):
        st.session_state.selected_runs.append(available_map[chosen])
        st.rerun()

    if not available:
        st.sidebar.info(f"No runs found in `{_RUNS_DIR}`.")

    # List selected runs with remove buttons
    if st.session_state.selected_runs:
        st.sidebar.markdown("---")
        st.sidebar.markdown("**Selected:**")
        for i, rp in enumerate(list(st.session_state.selected_runs)):
            label = Path(rp).name
            if st.sidebar.button(f"✕  {label}", key=f"rm_{i}", use_container_width=True):
                st.session_state.selected_runs.pop(i)
                st.rerun()

    # ----------------------------------------------------------------- empty state
    if not st.session_state.selected_runs:
        st.info("Add one or more runs from the sidebar to begin.")
        return

    # ----------------------------------------------------------------- load runs
    run_paths: list[str] = st.session_state.selected_runs
    runs: list[Run] = [_load_run(rp) for rp in run_paths]
    n = len(runs)

    # ----------------------------------------------------------------- shared controls
    min_steps = min(len(r.steps) for r in runs)
    min_layers = min(r.num_layers for r in runs)

    st.sidebar.markdown("---")
    st.sidebar.header("Controls")
    step = st.sidebar.selectbox("Step", list(range(min_steps)), index=0)
    layer = st.sidebar.slider("Layer", 0, max(min_layers - 1, 0), 0)
    source_options = [
        "hidden_out", "hidden_in", "q", "k", "v", "attn_out",
        "mlp_gate", "mlp_up", "mlp_act", "mlp_down_in", "mlp_down_out", "attn_weights",
    ]
    source = st.sidebar.selectbox("Heatmap source", source_options, index=0)
    cmap = st.sidebar.selectbox(
        "Colormap", ["RdBu_r", "viridis", "magma", "plasma", "Greys"], index=0
    )

    # ----------------------------------------------------------------- per-run header row
    header_cols = st.columns(n)
    for col, run, rp in zip(header_cols, runs, run_paths):
        s = run.summary()
        with col:
            st.subheader(Path(rp).name)
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("model", s["model"].split("/")[-1])
            m2.metric("layers", s["num_layers"])
            m3.metric("steps", s["num_steps"])
            m4.metric("device", s["device"])
            with st.expander("Prompt + generated"):
                st.write("**Prompt:**")
                st.code(s["prompt"], language="text")
                st.write("**Generated:**")
                st.code(s["generated"], language="text")

    # ----------------------------------------------------------------- tabs
    tab_heatmap, tab_residual, tab_logit, tab_neuron, tab_cross = st.tabs(
        ["Heatmap", "Residual stream", "Logit lens", "Neuron", "Cross-layer"]
    )

    # -- Heatmap
    with tab_heatmap:
        st.caption(f"{source} — layer {layer} @ step {step}")
        cols = st.columns(n)
        for col, run, rp in zip(cols, runs, run_paths, strict=False):
            with col:
                st.markdown(f"**{Path(rp).name}**")
                _render_heatmap(run, step, layer, source, cmap)

    # -- Residual stream
    with tab_residual:
        st.caption(f"step {step}")
        cols = st.columns(n)
        for col, run, rp in zip(cols, runs, run_paths, strict=False):
            with col:
                st.markdown(f"**{Path(rp).name}**")
                _render_residual(run, step)

    # -- Logit lens
    with tab_logit:
        lc1, lc2 = st.columns(2)
        position = lc1.number_input("Position (-1 = last)", value=-1, step=1)
        top_k_n = lc2.slider("top-k", 1, 20, 5)
        cols = st.columns(n)
        for i, (col, run, rp) in enumerate(zip(cols, runs, run_paths, strict=False)):
            with col:
                st.markdown(f"**{Path(rp).name}**")
                _render_logit_lens(run, step, int(position), top_k_n, btn_key=f"logit_{i}")

    # -- Neuron
    with tab_neuron:
        nc1, nc2 = st.columns(2)
        n_source = nc1.selectbox(
            "Neuron source",
            ["hidden_out", "mlp_act", "mlp_gate", "mlp_up", "attn_out"],
            key="neuron_src",
        )
        # Find max neuron dim across all runs so the slider is universal.
        max_D = 1
        for run in runs:
            try:
                probe = run.tensor(step, layer, n_source)
                max_D = max(max_D, probe.shape[-1] - 1)
            except KeyError:
                pass
        neuron = nc2.slider("Neuron index", 0, max_D, 0)
        cols = st.columns(n)
        for col, run, rp in zip(cols, runs, run_paths, strict=False):
            with col:
                st.markdown(f"**{Path(rp).name}**")
                _render_neuron(run, step, layer, n_source, neuron)

    # -- Cross-layer
    with tab_cross:
        cl_source = st.selectbox(
            "Source", ["hidden_out", "hidden_in", "attn_out", "mlp_down_out"], key="cl_src"
        )
        cols = st.columns(n)
        for col, run, rp in zip(cols, runs, run_paths, strict=False):
            with col:
                st.markdown(f"**{Path(rp).name}**")
                _render_cross_layer(run, step, cl_source)


if __name__ == "__main__":
    main()
