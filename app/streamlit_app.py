"""Simple leaf gallery: image, Healthy / Infected, and % when infected."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
LEAVES = ROOT / "outputs" / "leaves.csv"
FEATURES = ROOT / "outputs" / "features.csv"
INVENTORY = ROOT / "outputs" / "inventory.csv"
OOF = ROOT / "outputs" / "reports" / "oof_predictions.csv"
MODEL = ROOT / "outputs" / "models" / "leaf_model.joblib"
OVERLAYS = ROOT / "outputs" / "overlays"
THRESHOLD = 0.5

SCAN_LABELS = {
    "0dai_bi_visnir_bc_run2_1": "Before inoculation",
    "0dai_2hr_visnir_bc_run2-1": "2 hours after inoculation",
    "1dai_visnir_bc_run1": "Day 1",
    "2dai_visnir_bc_run1": "Day 2",
    "3dai_visnir_bc_run1": "Day 3",
}


@st.cache_resource
def _bundle() -> dict:
    return joblib.load(MODEL)


@st.cache_data(show_spinner=False)
def _tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    leaves = pd.read_csv(LEAVES)
    feats = pd.read_csv(FEATURES)
    inv = pd.read_csv(INVENTORY)
    bundle = _bundle()
    cols = list(bundle["feature_columns"])
    X = feats.reindex(columns=cols, fill_value=0.0).to_numpy(dtype=float)
    model = bundle["model"]
    if hasattr(model, "predict_proba"):
        p_fit = model.predict_proba(X)
        p_fit = p_fit[:, 1] if np.ndim(p_fit) == 2 else np.ravel(p_fit)
    else:
        p_fit = np.ravel(model.decision_function(X))
        p_fit = 1.0 / (1.0 + np.exp(-p_fit))
    feats = feats.copy()
    feats["P_show"] = p_fit
    if OOF.exists():
        oof = pd.read_csv(OOF)[["stem", "leaf_no", "P_infected"]]
        feats = feats.merge(oof, on=["stem", "leaf_no"], how="left")
        feats["P_show"] = feats["P_infected"].where(feats["P_infected"].notna(), feats["P_show"])
    leaves = leaves.merge(feats[["stem", "leaf_no", "P_show"]], on=["stem", "leaf_no"], how="left")
    return leaves, inv


def _is_infected(row: pd.Series) -> bool:
    tp = str(row.get("time_point", "")).lower()
    if tp in {"bi", "2hr"}:
        return False
    p = row.get("P_show")
    try:
        p = float(p)
    except (TypeError, ValueError):
        return False
    return bool(np.isfinite(p) and p >= THRESHOLD)


def _crop_leaf(overlay: Path, row: pd.Series, lines: int, samples: int, pad: int = 12) -> Image.Image | None:
    if not overlay.exists() or lines <= 0 or samples <= 0:
        return None
    im = Image.open(overlay).convert("RGB")
    ow, oh = im.size
    x0 = max(0, int(float(row["bbox_min_col"]) / samples * ow) - pad)
    y0 = max(0, int(float(row["bbox_min_row"]) / lines * oh) - pad)
    x1 = min(ow, int(float(row["bbox_max_col"]) / samples * ow) + pad)
    y1 = min(oh, int(float(row["bbox_max_row"]) / lines * oh) + pad)
    if x1 <= x0 or y1 <= y0:
        return im
    return im.crop((x0, y0, x1, y1))


def main() -> None:
    st.set_page_config(page_title="Tomato leaves", layout="wide")
    st.title("Tomato leaves")

    if not LEAVES.exists() or not MODEL.exists():
        st.info("Need outputs/leaves.csv and outputs/models/leaf_model.joblib.")
        return

    leaves, inv = _tables()
    size = {
        str(r.stem): (int(r.lines), int(r.samples))
        for r in inv.itertuples(index=False)
        if pd.notna(getattr(r, "lines", None))
    }
    stems = [s for s in leaves["stem"].dropna().unique().tolist()]
    options = ["All"] + [SCAN_LABELS.get(s, s) for s in stems]
    pick = st.selectbox("Scan", options=options, index=0)
    if pick == "All":
        work = leaves.copy()
    else:
        stem = next(s for s in stems if SCAN_LABELS.get(s, s) == pick)
        work = leaves[leaves["stem"] == stem].copy()
    work["infected"] = work.apply(_is_infected, axis=1)
    work = work.sort_values(["infected", "time_order", "leaf_no"], ascending=[True, True, True], na_position="last")

    def _cards(part: pd.DataFrame) -> None:
        if part.empty:
            return
        cols = st.columns(3)
        for i, (_, row) in enumerate(part.iterrows()):
            stem = str(row["stem"])
            lines, samples = size.get(stem, (1, 1))
            overlay = OVERLAYS / f"{stem}.png"
            infected = bool(row["infected"])
            with cols[i % 3]:
                crop = _crop_leaf(overlay, row, lines, samples)
                if crop is not None:
                    st.image(crop, width="stretch")
                scan_name = SCAN_LABELS.get(stem, stem)
                st.markdown(f"**Leaf {int(row['leaf_no'])}** · {scan_name}")
                if infected:
                    st.error("Infected")
                else:
                    st.success("Healthy")

    _cards(work[~work["infected"]])
    _cards(work[work["infected"]])


if __name__ == "__main__":
    main()
