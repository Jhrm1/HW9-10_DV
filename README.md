# DESI DR1: Quasar density on the sky

A multi-tab Streamlit app that maps **quasars per observed spectrum** on a HEALPix sky grid, built as the Week 10 assignment.

**Live app:** _paste your Streamlit Cloud URL here_
**Notebook:** `desi_sky_map.ipynb` — the record of how the app was built

---

## What this project is

The assignment asked for a layered thematic map on some partition of a space, with a spatial join, an attribute join, a rate, and a stated classification scheme — and two tabs. The reference implementation is US states. This project keeps that architecture and swaps the plane for the sky:

- **Regions** become HEALPix pixels at `nside=64` (NESTED), instead of US state polygons.
- **Points** become DESI DR1 quasars (`ra`, `dec`), instead of Tesla superchargers.
- **Denominator** becomes the count of all observed spectra per pixel, instead of an ACS population table.
- **Projection** becomes a celestial Mollweide, instead of Web Mercator.

Everything else — the two joins, the classification schemes, the sidebar filter, the coordinated view — carries over unchanged.

---

## Why a denominator at all

Quasars per pixel, plotted alone, is a map of where DESI pointed the telescope. The telescope does not look uniformly; it visits tiles on a schedule, and the exposure time varies with weather and survey phase. Dividing by **observed spectra per pixel** turns "how much DESI looked here" into "how many of the things DESI looked at were quasars" — which is a physical quantity, not an observational artefact.

That is why the attribute join is not optional in this project. It is what makes the choropleth mean anything.

---

## The data

Two CSVs, both pulled once from NOIRLab's Astro Data Lab TAP service and committed to the repo. The deployed app reads the files; it does not query the service.

| File | Columns | Source |
|---|---|---|
| `data/desi_qso.csv` | `ra`, `dec` | `desi_dr1.agnqso` (DESI DR1 AGN/QSO catalog) |
| `data/desi_pixel_totals.csv` | `healpix`, `n_spec` | `desi_dr1.zpix`, grouped by `healpix` |

Both tables are drawn from the same data release (DR1). A rate above 1 would mean they were not, and the notebook's final cell asserts against it.

**Refreshing the data.** Run cell 2 of `desi_sky_map.ipynb`. It writes both CSVs to `data/`.

---

## The two joins, and how they are shown

The rubric asks for both joins *and* a count of what did not match. Both are visible on the app's **"Both joins, shown"** tab.

### Spatial join — RA/Dec → HEALPix

```python
pix = np.full(len(qso), -1, dtype=int)
pix[good] = hp.ang2pix(NSIDE, ra[good], dec[good], lonlat=True, nest=True)
