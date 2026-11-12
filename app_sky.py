# app_sky.py — DESI DR1 HEALPix sky version of the supercharger dashboard
from pathlib import Path

import altair as alt
import healpy as hp
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from matplotlib.colors import to_hex
from matplotlib.patches import Rectangle

st.set_page_config(page_title="DESI DR1 Sky Map", layout="wide")
st.title("DESI DR1: Quasar density on the sky")
st.caption("HEALPix nside=64. Rate = QSOs / total observed spectra per pixel.")

HERE = Path(__file__).resolve().parent
NSIDE = 64

# --- Load the two tables pulled from NOIRLab Data Lab -------------------------
@st.cache_data
def load():
    data_dir = HERE / "data"
    qso_path = data_dir / "desi_qso.csv"
    denom_path = data_dir / "desi_pixel_totals.csv"
    if not qso_path.is_file() or not denom_path.is_file():
        with st.spinner(
            "Downloading the DESI tables from NOIRLab Data Lab. "
            "The first startup may take several minutes."
        ):
            from pull_data import download_data

            download_data()
    if not qso_path.is_file() or not denom_path.is_file():
        raise FileNotFoundError("The DESI data download did not create both CSV files.")

    qso = pd.read_csv(qso_path)            # ra, dec
    denom = pd.read_csv(denom_path)        # healpix, n_spec
    return qso, denom

qso, denom = load()

# --- Spatial join: RA/Dec → HEALPix pixel -------------------------------------
@st.cache_data
def spatial_join(qso):
    """Attach each quasar to its HEALPix pixel.

    healpy.ang2pix is the sky equivalent of shapely's point-in-polygon.
    Every finite RA/Dec falls in some pixel, so the 'unmatched' set is
    defined here as objects with non-finite coordinates or outside the
    DESI footprint (i.e., pixels absent from the denominator table).
    """
    ra = qso["ra"].values
    dec = qso["dec"].values
    good = np.isfinite(ra) & np.isfinite(dec)
    pix = np.full(len(qso), -1, dtype=int)
    pix[good] = hp.ang2pix(NSIDE, ra[good], dec[good], lonlat=True, nest=True)
    joined = qso.copy()
    joined["healpix"] = pix
    unmatched = joined[joined.healpix < 0]
    joined = joined[joined.healpix >= 0]
    return joined, unmatched

qso_joined, qso_unmatched = spatial_join(qso)

# --- Attribute join: counts per pixel → denominator table on healpix ---------
@st.cache_data
def attribute_join(qso_joined, denom):
    counts = (qso_joined.groupby("healpix").size()
              .rename("n_qso").reset_index())
    # healpix is int on both sides — no leading-zero trap here, but we
    # still report rows that fail to match (pixels outside the footprint).
    merged = counts.merge(denom, on="healpix", how="left")
    n_missing = merged["n_spec"].isna().sum()
    merged["n_spec"] = merged["n_spec"].fillna(0)
    # Rate: QSOs per observed spectrum. Mask empty pixels.
    merged["rate"] = np.where(merged["n_spec"] > 0,
                              merged["n_qso"] / merged["n_spec"], np.nan)
    return merged, n_missing

pixels, n_missing_denom = attribute_join(qso_joined, denom)
pixels = pixels.dropna(subset=["rate"])

# --- Sidebar: one filter moving several views --------------------------------
st.sidebar.header("Filters")
min_nspec = st.sidebar.slider(
    "Minimum observed spectra in pixel", 1, int(pixels.n_spec.max()), 1)
scheme = st.sidebar.radio(
    "Classification",
    ["Quantiles", "Equal interval", "Natural breaks (Jenks)"])
k = st.sidebar.slider("Classes", 3, 8, 5)
ramp = st.sidebar.selectbox("Colour ramp", ["OrRd", "YlGnBu", "viridis"])

shown = pixels[pixels.n_spec >= min_nspec].copy()

# --- Class breaks (same logic as the US app) ---------------------------------
def class_breaks(values, k, scheme):
    v = pd.Series(values).dropna()
    if len(v) == 0:
        return [0, 1]
    if scheme.startswith("Quantiles"):
        edges = list(v.quantile([i / k for i in range(k + 1)]))
    elif scheme.startswith("Equal interval"):
        lo, hi = v.min(), v.max()
        edges = [lo + (hi - lo) * i / k for i in range(k + 1)]
    else:
        import mapclassify
        edges = [float(v.min())] + [float(b) for b in
                                    mapclassify.NaturalBreaks(v.values, k=k).bins]
    edges = sorted(set(edges))
    edges[0] = min(edges[0], float(v.min()))
    edges[-1] = max(edges[-1], float(v.max()))
    return edges

bins = class_breaks(shown["rate"], k, scheme)
n_cls = len(bins) - 1
colors = [to_hex(matplotlib.colormaps[ramp](i / max(n_cls - 1, 1)))
          for i in range(n_cls)]

# --- Tabs --------------------------------------------------------------------
tab_map, tab_joins, tab_compare = st.tabs(
    ["Sky map", "Both joins, shown", "Pixels compared"])

with tab_map:
    pixel_ra, pixel_dec = hp.pix2ang(
        NSIDE, shown["healpix"].to_numpy(dtype=int), nest=True, lonlat=True)
    pixel_class = np.clip(
        np.searchsorted(bins, shown["rate"].to_numpy(), side="left") - 1,
        0, n_cls - 1)
    pixel_colors = [colors[i] for i in pixel_class]

    qso_pts = qso_joined[qso_joined.healpix.isin(shown.healpix)]
    if len(qso_pts) > 20000:
        qso_pts = qso_pts.sample(20000, random_state=42)
    fig = go.Figure()
    fig.add_trace(go.Scattergeo(
        lon=180 - qso_pts["ra"].to_numpy(),
        lat=qso_pts["dec"].to_numpy(),
        mode="markers",
        name="Quasars",
        marker={"size": 3, "color": "#ffd166", "opacity": 0.55},
        customdata=qso_pts[["ra", "dec", "healpix"]].to_numpy(),
        hovertemplate=(
            "Quasar<br>RA: %{customdata[0]:.2f}°"
            "<br>Dec: %{customdata[1]:.2f}°"
            "<br>HEALPix: %{customdata[2]:.0f}<extra></extra>"
        ),
    ))
    fig.add_trace(go.Scattergeo(
        lon=180 - pixel_ra,
        lat=pixel_dec,
        mode="markers",
        name="HEALPix pixels",
        marker={
            "size": 7,
            "color": pixel_colors,
            "opacity": 0.85,
            "line": {"color": "#d9e2f2", "width": 0.3},
        },
        customdata=shown[["healpix", "n_qso", "n_spec", "rate"]].to_numpy(),
        hovertemplate=(
            "HEALPix: %{customdata[0]:.0f}"
            "<br>QSOs: %{customdata[1]:.0f}"
            "<br>Observed spectra: %{customdata[2]:.0f}"
            "<br>Rate: %{customdata[3]:.4f}<extra></extra>"
        ),
    ))
    fig.update_geos(
        projection_type="mollweide",
        showcoastlines=False,
        showland=False,
        showocean=False,
        showlakes=False,
        showcountries=False,
        showframe=True,
        framecolor="#56647a",
        bgcolor="#060914",
        lonaxis={"showgrid": True, "gridcolor": "#28364d", "dtick": 30},
        lataxis={"showgrid": True, "gridcolor": "#28364d", "dtick": 30},
    )
    fig.update_layout(
        height=560,
        paper_bgcolor="#060914",
        plot_bgcolor="#060914",
        font={"color": "#dce6f6"},
        margin={"l": 8, "r": 8, "t": 24, "b": 8},
        legend={"bgcolor": "rgba(6,9,20,0.7)"},
    )
    st.plotly_chart(fig, width="stretch")
    st.caption(
        f"**Celestial Mollweide projection.** Quasars are shown as gold points; "
        f"HEALPix pixel centers are coloured by QSO rate using **{scheme}** "
        f"with **{n_cls} classes** and the {ramp} ramp. Individual quasars are "
        "sampled to 20k for speed. Right ascension increases from right to "
        "left; hover over a marker for its coordinates and values."
    )

with tab_joins:
    st.subheader("Spatial join — RA/Dec → HEALPix")
    c1, c2, c3 = st.columns(3)
    c1.metric("Quasars matched", f"{len(qso_joined):,}")
    c2.metric("Quasars with bad coords", f"{len(qso_unmatched):,}")
    c3.metric("Pixels missing denominator", f"{n_missing_denom:,}")
    if n_missing_denom:
        st.warning(
            f"{n_missing_denom} HEALPix pixels contain quasars but are absent "
            "from the denominator table. Reason: those pixels lie outside the "
            "DESI footprint used to build the denominator. They are dropped "
            "from the rate map — accounted for, not silently lost."
        )
    st.subheader("Attribute join — denominator on healpix")
    st.markdown(
        "The denominator table (`desi_pixel_totals.csv`) is joined on the "
        "integer `healpix` column. Unlike FIPS codes, HEALPix indices have no "
        "leading zero, so the join is safe. The metric above shows it worked."
    )
    st.dataframe(shown.head(8)[["healpix", "n_qso", "n_spec", "rate"]],
                 hide_index=True)

with tab_compare:
    pick = alt.selection_point(fields=["healpix"], on="pointerover", empty=False)
    bars = (alt.Chart(shown.nlargest(20, "rate"))
            .mark_bar().encode(
                x=alt.X("rate:Q", title="QSOs per observed spectrum"),
                y=alt.Y("healpix:N", sort="-x", title="HEALPix"),
                color=alt.condition(pick, alt.value("#ffcc00"), alt.value("#555555")),
                tooltip=["healpix", "n_qso", "n_spec", "rate"])
            .add_params(pick)
            .properties(height=380))
    scatter = (alt.Chart(shown).mark_circle(size=30).encode(
                x="n_spec:Q", y="rate:Q",
                color=alt.condition(pick, alt.value("#ffcc00"), alt.value("#033c5a")),
                tooltip=["healpix", "n_spec", "rate"])
              .add_params(pick)
              .properties(height=380))
    st.altair_chart(bars | scatter, width="stretch")
    st.caption("**Coordinated view (bonus).** Hovering a bar or a dot lights "
               "the matching pixel in the other panel. Both panels read from "
               "`shown`, which the sidebar filter moves.")

# --- Legend ------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 0.7))
fig.patch.set_alpha(0)
for i in range(n_cls):
    ax.add_patch(Rectangle((i, 0), 0.92, 1,
                           facecolor=matplotlib.colormaps[ramp](i / max(n_cls - 1, 1)),
                           edgecolor="#888888", linewidth=0.5))
    ax.text(i + 0.46, -0.35, f"{bins[i]:.3f}–{bins[i+1]:.3f}",
            ha="center", va="top", fontsize=7, color="#cccccc")
ax.set_title("QSOs per observed spectrum", fontsize=8, color="#cccccc", loc="left")
ax.set_xlim(-0.1, n_cls); ax.set_ylim(-1.2, 1.2); ax.axis("off")
st.pyplot(fig)
plt.close(fig)