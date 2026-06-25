"""Live demo dashboard for the BOSCH Plasma-Etching Virtual Metrology model.

    streamlit run demo/app.py

Reads the precomputed demo/bundle/ (see demo/build_bundle.py). oxide_etch is the
LIVE centerpiece — pressing "추론 실행" runs the genuine Cycle-Aware DL model
forward on the selected held-out wafer's sensor tensors and reveals the predicted
wafer map. si_etch is shown from precomputed predictions as spatial-dominant
context (it motivates why oxide is the real VM contribution).

The story the dashboard tells:
  센서(OES+Process) → 딥러닝 모델 → 89-point 웨이퍼 etch 맵
  그리고 그 정확도를 XGBoost / Spatial-mean baseline 과 같은 자리에서 비교.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

# Korean glyphs in plot labels need a CJK font (Windows ships Malgun Gothic).
try:
    import matplotlib.font_manager as _fm
    _avail = {f.name for f in _fm.fontManager.ttflist}
    for _f in ("Malgun Gothic", "NanumGothic", "Gulim", "Batang"):
        if _f in _avail:
            # CJK font first, DejaVu Sans as per-glyph fallback (e.g. minus sign).
            plt.rcParams["font.family"] = [_f, "DejaVu Sans"]
            break
except Exception:
    pass
plt.rcParams["axes.unicode_minus"] = False

from src.demo import DLPredictor, ModelInputs

BUNDLE = PROJECT_ROOT / "demo" / "bundle"

st.set_page_config(page_title="Plasma-Etching VM Demo", page_icon="🔬", layout="wide")


# ─────────────────────────────────────────────────────────────────────────────
# Loaders (cached)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_manifest() -> dict:
    return json.loads((BUNDLE / "manifest.json").read_text(encoding="utf-8"))


@st.cache_data(show_spinner=False)
def load_wafer(key: str) -> dict:
    z = np.load(BUNDLE / "wafers" / f"{key}.npz")
    return {k: z[k] for k in z.files}


@st.cache_resource(show_spinner=False)
def load_oxide_predictor(ckpt_rel: str) -> DLPredictor:
    pred = DLPredictor.from_checkpoint(BUNDLE / ckpt_rel)
    return pred


def run_oxide_live(pred: DLPredictor, w: dict) -> tuple[np.ndarray, float]:
    """Run the genuine model forward on stored (normalized) inputs. Returns (pred, seconds)."""
    inputs = ModelInputs(oes=w["ox_oes"], proc=w["ox_proc"], xy=w["ox_xy"])
    # warm up CUDA kernels once so the displayed latency is steady-state
    pred.forward(inputs)
    t0 = time.perf_counter()
    out = pred.forward(inputs)
    return out, time.perf_counter() - t0


# ─────────────────────────────────────────────────────────────────────────────
# Plot helpers (English labels → no matplotlib CJK-font issues)
# ─────────────────────────────────────────────────────────────────────────────
def wafer_scatter(ax, X, Y, vals, *, vmin, vmax, cmap, title, unit, s=230, cbar=True):
    r = float(np.max(np.sqrt(X**2 + Y**2))) * 1.08
    circ = plt.Circle((0, 0), r, fill=False, ls="--", lw=1.0, color="0.6")
    ax.add_patch(circ)
    sc = ax.scatter(X, Y, c=vals, cmap=cmap, vmin=vmin, vmax=vmax,
                    s=s, edgecolors="white", linewidths=0.4, zorder=3)
    ax.set_aspect("equal")
    ax.set_xlim(-r * 1.05, r * 1.05)
    ax.set_ylim(-r * 1.05, r * 1.05)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(title, fontsize=13, fontweight="bold")
    if cbar:
        cb = ax.figure.colorbar(sc, ax=ax, fraction=0.046, pad=0.03)
        cb.set_label(unit, fontsize=9)
    return sc


def triptych(X, Y, y_true, y_pred, unit):
    # Predicted + Ground-truth share ONE colorbar (same scale) so the wafer
    # circles can be drawn large; Error gets its own diverging colorbar.
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 5.0), constrained_layout=True)
    vmin, vmax = float(np.min(y_true)), float(np.max(y_true))
    wafer_scatter(axes[0], X, Y, y_pred, vmin=vmin, vmax=vmax, cmap="viridis",
                  title="Predicted (VM model)", unit=unit, cbar=False)
    sc = wafer_scatter(axes[1], X, Y, y_true, vmin=vmin, vmax=vmax, cmap="viridis",
                       title="Ground truth (physical)", unit=unit, cbar=False)
    fig.colorbar(sc, ax=axes[:2], fraction=0.025, pad=0.01, location="right").set_label(unit, fontsize=9)
    resid = y_pred - y_true
    amax = float(np.max(np.abs(resid))) or 1e-6
    scE = wafer_scatter(axes[2], X, Y, resid, vmin=-amax, vmax=amax, cmap="RdBu_r",
                        title="Error  (pred − true)", unit=unit, cbar=False)
    fig.colorbar(scE, ax=axes[2], fraction=0.05, pad=0.01).set_label(unit, fontsize=9)
    return fig


def truth_only(X, Y, y_true, unit):
    fig, ax = plt.subplots(figsize=(5.0, 4.6), constrained_layout=True)
    wafer_scatter(ax, X, Y, y_true, vmin=float(np.min(y_true)), vmax=float(np.max(y_true)),
                  cmap="viridis", title="Ground truth — model not run yet", unit=unit)
    return fig


def oes_preview(ox_oes: np.ndarray):
    """Heatmap of what the model 'sees': band-selected OES, mean over time → (100 cycles × 256 bands)."""
    img = ox_oes.mean(axis=1).T  # (256 bands, 100 cycles)
    fig, ax = plt.subplots(figsize=(6.4, 3.1), constrained_layout=True)
    im = ax.imshow(img, aspect="auto", cmap="magma", origin="lower")
    ax.set_xlabel("BOSCH cycle (1 → 100)", fontsize=9)
    ax.set_ylabel("OES band", fontsize=9)
    ax.tick_params(labelsize=8)
    ax.set_title("Model input — OES × 100 cycles (normalized)", fontsize=11, fontweight="bold")
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    return fig


def comparison_bar(wm):
    labels = ["Cycle-Aware DL\n(제안 모델)", "XGBoost\nbaseline", "Spatial-mean\nbaseline"]
    r2s = [wm["dl"]["r2"], wm["xgb"]["r2"], wm["spatial"]["r2"]]
    colors = ["#1976D2", "#E64A19", "#90A4AE"]
    fig, ax = plt.subplots(figsize=(6.4, 3.6), constrained_layout=True)
    bars = ax.bar(labels, r2s, color=colors, edgecolor="black", linewidth=0.5, width=0.62)
    ax.axhline(0, color="0.4", lw=0.8)
    ax.set_ylabel("R²  (이 웨이퍼)", fontsize=10)
    ax.tick_params(labelsize=9)
    ax.set_ylim(min(0, min(r2s)) - 0.14, 1.08)
    for b, v in zip(bars, r2s):
        ax.text(b.get_x() + b.get_width() / 2, v + (0.03 if v >= 0 else -0.1),
                f"{v:.2f}", ha="center", fontsize=12, fontweight="bold")
    ax.grid(axis="y", ls="--", alpha=0.4)
    return fig


def metrics_of(y_true, y_pred) -> dict:
    from src.evaluation import regression_metrics
    return regression_metrics(y_true, y_pred)


# ─────────────────────────────────────────────────────────────────────────────
# App
# ─────────────────────────────────────────────────────────────────────────────
if not (BUNDLE / "manifest.json").exists():
    st.error("demo/bundle/ 이 없습니다. 먼저 번들을 생성하세요:\n\n"
             "`.venv\\python.exe -m demo.build_bundle`")
    st.stop()

man = load_manifest()

# Compact spacing + hide Streamlit's top toolbar so nothing fits one laptop
# screen without scrolling and the title is never clipped.
st.markdown(
    "<style>"
    "header[data-testid='stHeader']{display:none;}"
    ".block-container{padding-top:1.4rem;padding-bottom:0.4rem;max-width:100%;}"
    "[data-testid='stMetric']{padding:0;} [data-testid='stMetricValue']{font-size:1.6rem;}"
    "[data-testid='stImage'],[data-testid='stImage'] img{margin:0 auto;}"
    "h4{margin-bottom:0.1rem;} hr{margin:0.3rem 0;}"
    "</style>",
    unsafe_allow_html=True,
)
st.markdown("#### 🔬 플라즈마 식각 가상계측 (Virtual Metrology) — Live Demo")

# ── sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ 시연 설정")
    per_wafer = {w["key"]: w for w in man["per_wafer"]}
    def _label(k):
        w = per_wafer[k]
        return f"{k}  (lot {w['lot']})"
    wafer_key = st.selectbox("테스트 웨이퍼 (held-out, fold 0)", man["wafers"],
                             format_func=_label)
    target = st.radio(
        "예측 타겟",
        ["oxide_etch", "si_etch"],
        format_func=lambda t: ("oxide_etch — 본 연구 핵심 (process-driven)"
                               if t == "oxide_etch"
                               else "si_etch — 공간 지배 (spatial, 참고)"),
    )
    show_oes = st.checkbox("모델 입력(OES 스펙트럼) 미리보기", value=True)
    st.divider()
    st.caption("OES + Process 센서 신호만으로 웨이퍼 89개 지점의 식각량을 예측합니다. "
               "물리 계측은 파괴·고비용·수십 분 — VM은 비파괴로 즉시 추정합니다.")
    st.markdown(f"**검증 데이터** · 웨이퍼 {man['n_wafers']}개 × 89지점 · "
                f"`{man['split_file']}` fold {man['fold']} · 전부 held-out")

w = load_wafer(wafer_key)
X, Y = w["X"], w["Y"]
unit = "μm"
is_oxide = target == "oxide_etch"
y_true = w["oxide_true"] if is_oxide else w["si_true"]

# ── headline / pooled KPI row ────────────────────────────────────────────────
pool = man["pooled"]["oxide" if is_oxide else "si"]
c1, c2, c3, c4 = st.columns(4)
c1.metric("전체 검증 R² (DL)", f"{pool['dl']['r2']:.3f}",
          help="번들 18개 웨이퍼 전체 포인트 기준 (논문 헤드라인 수치)")
c2.metric("XGBoost baseline", f"{pool['xgb']['r2']:.3f}",
          delta=f"{pool['dl']['r2'] - pool['xgb']['r2']:+.3f}")
c3.metric("Spatial-mean baseline", f"{pool['spatial']['r2']:.3f}",
          delta=f"{pool['dl']['r2'] - pool['spatial']['r2']:+.3f}")
c4.metric("DL 추론 시간", "수십 ms", help="물리 계측 수십 분 → VM은 즉시·비파괴")


def _metric_line(m: dict, secs: float | None = None) -> str:
    s = (f"이 웨이퍼 →  **R² {m['r2']:.3f}**  ·  RMSE {m['rmse']:.4f} μm  ·  MAE {m['mae']:.4f} μm")
    if secs is not None:
        s += f"  ·  ⏱ **{secs*1000:.0f} ms**"
    return s


# ── two-column main: wafer maps (left) | OES input + model compare (right) ────
left, right = st.columns([1.7, 1], gap="medium")

with left:
    st.markdown(f"**② 추론 결과 — `{target}`**")
    if is_oxide:
        state_key = f"ox_pred::{wafer_key}"
        run = st.button("🚀 모델 추론 실행 (Run inference)", type="primary",
                        use_container_width=True)
        if run:
            pred = load_oxide_predictor(man["checkpoint_oxide"])
            with st.spinner("Cycle-Aware DL 모델 추론 중..."):
                y_pred, secs = run_oxide_live(pred, w)
            st.session_state[state_key] = (y_pred, secs)

        if state_key in st.session_state:
            y_pred, secs = st.session_state[state_key]
            m = metrics_of(y_true, y_pred)
            st.markdown(_metric_line(m, secs))
            st.pyplot(triptych(X, Y, y_true, y_pred, unit), use_container_width=True)
        else:
            st.info("▶ 버튼을 누르면 센서 신호를 모델에 통과시켜 실시간 추론합니다. "
                    "(현재는 실측 맵만 표시)")
            _c = st.columns([1, 2, 1])
            _c[1].pyplot(truth_only(X, Y, y_true, unit), use_container_width=True)
    else:
        y_pred = w["si_pred_dl"]   # precomputed (si model needs full 3648-band OES)
        m = metrics_of(y_true, y_pred)
        st.markdown(_metric_line(m))
        st.pyplot(triptych(X, Y, y_true, y_pred, unit), use_container_width=True)
        st.caption("si_etch는 **공간 지배적** (위치만으로 R²≈0.98) → 진짜 도전은 oxide_etch. "
                   f"※ si는 초기 single-fold 모델({man['models']['si']['exp'].split('/')[-1]}) 결과이며, "
                   "최종 best 스택(aux+mixup+EMA)은 oxide에 적용되어 라이브 시연됩니다.")

with right:
    if show_oes:
        st.markdown("**① 모델 입력 — OES 스펙트럼**")
        st.pyplot(oes_preview(w["ox_oes"]), use_container_width=True)
    st.markdown("**③ 같은 웨이퍼, 세 모델 비교**")
    st.pyplot(comparison_bar(per_wafer[wafer_key]["oxide" if is_oxide else "si"]),
              use_container_width=True)

# ── bottom takeaway bar (uses the lower space, reinforces the message) ────────
d_xgb = pool["dl"]["r2"] - pool["xgb"]["r2"]
st.success(
    f"💡 **VM의 가치** — 물리 계측(파괴·수십 분)을 센서 신호 기반 **비파괴 추론(수십 ms)** 으로 대체. "
    f"핵심 타겟 oxide_etch 에서 제안 모델이 전체 검증 R² **{pool['dl']['r2']:.3f}** 로 "
    f"XGBoost(**{pool['xgb']['r2']:.3f}**, +{d_xgb:.3f}) · Spatial-mean(**{pool['spatial']['r2']:.3f}**) baseline 을 모두 상회."
    if is_oxide else
    f"💡 si_etch 는 공간 지배적이라 baseline 도 R²≈0.98 — **본 연구의 기여는 공정 동역학이 결정하는 oxide_etch** 에서 드러납니다."
)
