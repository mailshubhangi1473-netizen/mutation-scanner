import re

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from Bio.SeqUtils import molecular_weight
from Bio.SeqUtils.ProtParam import ProteinAnalysis
from Bio.SeqUtils.ProtParamData import kd

from scanner import (AA, CHARGE, DEFAULT_WEIGHTS, VOLUME, disruption_score,
                     mutation_table, parse_sequence, scan,
                     substitution_components)

st.set_page_config(page_title="Mutation Scanner", page_icon="🧬", layout="wide")

# ------------------------------------------------------------------ design tokens
INK, PAPER, MUTED = "#1B2433", "#F7F9FB", "#5F6B7A"
# sensitivity ramp: pale ice -> steel blue -> orchid -> deep plum
STOPS = [(0.0, (243, 246, 249)), (0.35, (157, 191, 214)), (0.65, (199, 91, 138)), (1.0, (90, 19, 70))]
PLOTLY_SCALE = [[p, f"rgb{c}"] for p, c in STOPS]
# rows grouped by chemistry so patterns read at a glance
ROW_ORDER = "AVILMFWC" + "GPSTYNQH" + "DEKR"
EXAMPLE = ("MALWMRLLPLLALLALWGPDPAAAFVNQHLCGSHLVEALYLVCGERGFFYTPKTRREAEDLQVGQVELGGGPGAGSLQPLALEGSLQKRGIVEQCCTSICSLYQLENYCN")


def color_at(t: float) -> str:
    t = float(np.clip(t, 0, 1))
    for (p0, c0), (p1, c1) in zip(STOPS, STOPS[1:]):
        if t <= p1:
            f = (t - p0) / (p1 - p0)
            r, g, b = (round(a + (b_ - a) * f) for a, b_ in zip(c0, c1))
            return f"rgb({r},{g},{b})"
    return f"rgb{STOPS[-1][1]}"


st.markdown(
    f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;600;800&family=JetBrains+Mono:wght@500&display=swap');
html, body, [class*="css"], .stMarkdown, .stTextArea textarea {{ font-family: 'Manrope', sans-serif; }}
.block-container {{ padding-top: 2.2rem; max-width: 1250px; }}
h1.title {{ font-weight: 800; font-size: 2.6rem; letter-spacing: -0.03em; margin: 0; color: {INK}; }}
p.lede {{ color: {MUTED}; font-size: 1.05rem; margin: .3rem 0 1.4rem; max-width: 46rem; }}
.stats {{ display: flex; gap: 0; margin: 1.2rem 0 1.6rem; border-top: 2px solid {INK}; }}
.stat {{ flex: 1; padding: .7rem 1rem .2rem 0; }}
.stat + .stat {{ padding-left: 1rem; border-left: 1px solid #D5DCE4; }}
.stat .v {{ font-size: 1.7rem; font-weight: 800; letter-spacing: -0.02em; color: {INK}; }}
.stat .l {{ font-size: .82rem; color: {MUTED}; }}
.strip {{ background: #fff; border: 1px solid #D5DCE4; border-radius: 10px; padding: 14px 16px; overflow-x: auto; }}
.strip .line {{ display: flex; align-items: center; margin: 3px 0; white-space: nowrap; }}
.strip .ln {{ width: 3.2rem; color: {MUTED}; font: 500 .75rem 'JetBrains Mono', monospace; text-align: right; padding-right: 12px; }}
.strip .r {{ display: inline-block; width: 1.5rem; text-align: center; padding: 4px 0;
  font: 500 .88rem 'JetBrains Mono', monospace; }}
.strip .r.gap {{ margin-right: 10px; }}
.legend {{ display: flex; align-items: center; gap: 10px; font-size: .8rem; color: {MUTED}; margin: .6rem 0 0; }}
.legend .bar {{ height: 10px; width: 180px; border-radius: 5px;
  background: linear-gradient(90deg, {color_at(0)}, {color_at(.35)}, {color_at(.65)}, {color_at(1)}); }}
.verdict {{ font-size: 2.4rem; font-weight: 800; letter-spacing: -0.03em; margin: 0; }}
.verdict small {{ font-size: 1rem; font-weight: 600; color: {MUTED}; margin-left: .6rem; }}
section[data-testid="stSidebar"] {{ border-right: 1px solid #D5DCE4; }}
button[data-baseweb="tab"] {{ font-weight: 600; }}
</style>
""",
    unsafe_allow_html=True,
)

# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.markdown("### Scoring weights")
    st.caption("Choose how much each property counts toward a substitution's disruption score. Only the ratios matter.")
    weights = {
        "hydrophobicity": st.slider("Hydrophobicity", 0.0, 1.0, DEFAULT_WEIGHTS["hydrophobicity"], 0.05,
                                    help="Kyte-Doolittle scale"),
        "charge": st.slider("Charge", 0.0, 1.0, DEFAULT_WEIGHTS["charge"], 0.05, help="Side-chain charge at pH 7.4"),
        "volume": st.slider("Side-chain size", 0.0, 1.0, DEFAULT_WEIGHTS["volume"], 0.05, help="Residue volume in Å³"),
        "blosum62": st.slider("Evolutionary similarity", 0.0, 1.0, DEFAULT_WEIGHTS["blosum62"], 0.05,
                              help="BLOSUM62: how rarely this swap is seen in related proteins"),
    }
    if sum(weights.values()) == 0:
        st.error("Set at least one weight above zero.")
        st.stop()
    if st.button("Reset weights", width="stretch"):
        st.rerun()
    st.divider()
    st.caption("Scores run from 0 (no change) to 1 (most disruptive possible). This is a physicochemical screen. "
               "It ignores 3D structure and conservation, so confirm hits with SIFT, PolyPhen or AlphaMissense.")

# ------------------------------------------------------------------ header + input
st.markdown('<h1 class="title">Mutation scanner</h1>', unsafe_allow_html=True)
st.markdown(
    '<p class="lede">Find the residues a protein cannot afford to change. Paste a sequence and score all '
    "19 possible substitutions at every position.</p>",
    unsafe_allow_html=True,
)

if "seq_text" not in st.session_state:
    st.session_state["seq_text"] = ""

left, right = st.columns([3, 1], gap="large")
with right:
    uploaded = st.file_uploader("Upload FASTA", type=["fasta", "fa", "faa", "txt"], label_visibility="collapsed")
    if uploaded is not None and st.session_state.get("_last_upload") != uploaded.file_id:
        st.session_state["seq_text"] = uploaded.read().decode("utf-8", errors="ignore")
        st.session_state["_last_upload"] = uploaded.file_id
        st.rerun()
    if st.button("Try an example (human insulin)", width="stretch"):
        st.session_state["seq_text"] = EXAMPLE
        st.rerun()
with left:
    text = st.text_area("Protein sequence", key="seq_text", height=110,
                        placeholder="Paste a raw amino-acid sequence or FASTA here")

if not text.strip():
    st.info("Paste a protein sequence, upload a FASTA file, or try the example to get started.")
    st.stop()

try:
    name, seq = parse_sequence(text)
except ValueError as e:
    st.error(f"{e}")
    st.stop()


@st.cache_data(show_spinner=False)
def run_scan(seq: str, w: tuple):
    return scan(seq, dict(w))


matrix, positions = run_scan(seq, tuple(weights.items()))
pa = ProteinAnalysis(seq)
st.markdown(
    f"""<div class="stats">
<div class="stat"><div class="v">{len(seq)}</div><div class="l">residues</div></div>
<div class="stat"><div class="v">{molecular_weight(seq, 'protein') / 1000:.1f} kDa</div><div class="l">molecular weight</div></div>
<div class="stat"><div class="v">{pa.isoelectric_point():.2f}</div><div class="l">isoelectric point</div></div>
<div class="stat"><div class="v">{len(seq) * 19:,}</div><div class="l">variants scored</div></div>
</div>""",
    unsafe_allow_html=True,
)

tab_map, tab_hot, tab_var, tab_data = st.tabs(["Sensitivity map", "Hotspots", "Check a mutation", "Export"])

# ------------------------------------------------------------------ tab 1: strip + heatmap
with tab_map:
    st.markdown("**Your sequence, shaded by sensitivity**")
    vals = positions["mean_disruption"].values
    lo, hi = vals.min(), vals.max()
    per_line = 50
    parts = ['<div class="strip">']
    for start in range(0, len(seq), per_line):
        parts.append(f'<div class="line"><span class="ln">{start + 1}</span><span>')
        for i in range(start, min(start + per_line, len(seq))):
            t = (vals[i] - lo) / (hi - lo) if hi > lo else 0.0
            fg = "#fff" if t > 0.55 else INK
            gap = " gap" if (i - start) % 10 == 9 else ""
            parts.append(
                f'<span class="r{gap}" style="background:{color_at(t)};color:{fg}" '
                f'title="{seq[i]}{i + 1}: mean disruption {vals[i]:.2f}">{seq[i]}</span>'
            )
        parts.append("</span></div>")
    parts.append("</div>")
    parts.append(
        '<div class="legend">Tolerant<span class="bar"></span>Sensitive'
        "<span>(shaded relative to this protein; hover a residue for its score)</span></div>"
    )
    st.markdown("".join(parts), unsafe_allow_html=True)

    st.markdown("&nbsp;")
    st.markdown("**Every substitution**")
    window = 80
    if len(seq) > window:
        start = st.slider("Window start", 1, len(seq) - window + 1, 1, help="Scroll along the protein")
    else:
        start = 1
    end = min(start + window - 1, len(seq))
    cols = list(range(start, end + 1))
    sub = matrix.loc[list(ROW_ORDER), cols]
    custom = [[f"{seq[p - 1]}{p}{m}" for p in cols] for m in ROW_ORDER]
    fig = go.Figure(
        go.Heatmap(
            z=sub.values,
            x=[f"{seq[p - 1]}{p}" for p in cols],
            y=list(ROW_ORDER),
            customdata=custom,
            colorscale=PLOTLY_SCALE,
            zmin=0, zmax=1,
            xgap=1, ygap=1,
            hovertemplate="%{customdata}<br>disruption %{z:.2f}<extra></extra>",
            colorbar=dict(title="Disruption", thickness=12, len=0.8),
        )
    )
    for y in (7.5, 15.5):  # separate hydrophobic / polar / charged groups
        fig.add_hline(y=y, line_color=INK, line_width=1.5)
    fig.update_yaxes(autorange="reversed", title="Mutant residue")
    fig.update_xaxes(tickangle=-90, tickfont=dict(size=9), side="top")
    fig.update_layout(height=560, margin=dict(l=10, r=10, t=70, b=10), plot_bgcolor=PAPER,
                      paper_bgcolor=PAPER, font=dict(family="Manrope", color=INK))
    st.plotly_chart(fig, width="stretch")
    st.caption(f"Positions {start}-{end}. Rows are grouped: hydrophobic, polar, then charged residues. "
               "A dark column means almost any change there is damaging.")

# ------------------------------------------------------------------ tab 2: hotspots
with tab_hot:
    c1, c2 = st.columns([2, 1], gap="large")
    with c1:
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=positions["position"], y=positions["mean_disruption"], mode="lines",
            line=dict(color="#B0436F", width=2), fill="tozeroy", fillcolor="rgba(176,67,111,0.12)",
            customdata=positions["residue"], hovertemplate="%{customdata}%{x}: %{y:.2f}<extra></extra>"))
        fig2.update_layout(height=360, margin=dict(l=10, r=10, t=10, b=10), plot_bgcolor=PAPER,
                           paper_bgcolor=PAPER, font=dict(family="Manrope", color=INK),
                           xaxis_title="Position", yaxis_title="Mean disruption", yaxis_range=[0, 1])
        st.plotly_chart(fig2, width="stretch")
    with c2:
        top_n = st.slider("Positions to list", 5, 30, 10)
        st.caption("Ranked by mean disruption across all 19 substitutions.")
    top = positions.sort_values("mean_disruption", ascending=False).head(top_n).reset_index(drop=True)
    st.dataframe(
        top, width="stretch", hide_index=True,
        column_config={
            "position": st.column_config.NumberColumn("Position", format="%d"),
            "residue": "Residue",
            "mean_disruption": st.column_config.ProgressColumn("Mean disruption", min_value=0, max_value=1, format="%.2f"),
            "max_disruption": st.column_config.ProgressColumn("Worst substitution", min_value=0, max_value=1, format="%.2f"),
        },
    )

# ------------------------------------------------------------------ tab 3: single mutation
with tab_var:
    st.markdown("Type a mutation in standard notation: wild-type letter, position, new letter (for example "
                f"**{seq[0]}1{'A' if seq[0] != 'A' else 'G'}**).")
    q = st.text_input("Mutation", placeholder=f"{seq[min(11, len(seq) - 1)]}{min(12, len(seq))}A",
                      label_visibility="collapsed").strip().upper()
    if not q:
        st.caption("Enter a mutation to see how it compares with every other change in this protein.")
    else:
        m = re.fullmatch(r"([A-Z])(\d+)([A-Z])", q)
        if not m:
            st.error("Use the format G12D: one letter, a position number, one letter.")
        else:
            wt, pos, mut = m.group(1), int(m.group(2)), m.group(3)
            if not 1 <= pos <= len(seq):
                st.error(f"Position {pos} is outside the sequence (1-{len(seq)}).")
            elif seq[pos - 1] != wt:
                st.error(f"Position {pos} is {seq[pos - 1]} in this sequence, not {wt}.")
            elif mut not in AA or mut == wt:
                st.error("The new residue must be a different standard amino acid.")
            else:
                score = disruption_score(wt, mut, weights)
                all_scores = matrix.values[~np.isnan(matrix.values)]
                pct = float((all_scores < score).mean() * 100)
                st.markdown(f'<p class="verdict">{q}<small>disruption {score:.2f}</small></p>', unsafe_allow_html=True)
                st.markdown(f"More disruptive than **{pct:.0f}%** of all {len(all_scores):,} possible substitutions in this protein.")

                a, b = st.columns([1, 1], gap="large")
                with a:
                    comps = substitution_components(wt, mut)
                    labels = {"hydrophobicity": "Hydrophobicity", "charge": "Charge",
                              "volume": "Side-chain size", "blosum62": "Evolutionary similarity"}
                    figc = go.Figure(go.Bar(
                        x=list(comps.values()), y=[labels[k] for k in comps], orientation="h",
                        marker_color=[color_at(v) for v in comps.values()],
                        text=[f"{v:.2f}" for v in comps.values()], textposition="outside"))
                    figc.update_xaxes(range=[0, 1.15], title="Scaled change (0 to 1)")
                    figc.update_yaxes(autorange="reversed")
                    figc.update_layout(height=280, margin=dict(l=10, r=10, t=10, b=10), plot_bgcolor=PAPER,
                                       paper_bgcolor=PAPER, font=dict(family="Manrope", color=INK))
                    st.plotly_chart(figc, width="stretch")
                with b:
                    st.dataframe(
                        pd.DataFrame({
                            "": ["Hydropathy", "Charge (pH 7.4)", "Volume (Å³)"],
                            wt + " (original)": [kd[wt], CHARGE[wt], VOLUME[wt]],
                            mut + " (mutant)": [kd[mut], CHARGE[mut], VOLUME[mut]],
                        }),
                        hide_index=True, width="stretch",
                    )
                    st.caption(f"Every substitution at position {pos}, most to least disruptive:")
                    row = matrix[pos].dropna().sort_values(ascending=False)
                    st.write("  ".join(f"{wt}{pos}{k}" for k in row.index[:6]) + " …")

# ------------------------------------------------------------------ tab 4: export
with tab_data:
    table = mutation_table(seq, weights)
    st.markdown("All variants, ranked from most to least disruptive.")
    st.dataframe(
        table.head(300), width="stretch", hide_index=True,
        column_config={"score": st.column_config.ProgressColumn("Disruption", min_value=0, max_value=1, format="%.3f")},
    )
    st.download_button("Download all variants (CSV)", table.to_csv(index=False).encode(),
                       file_name=f"{name}_mutation_scan.csv", mime="text/csv", type="primary")
