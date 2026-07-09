"""SHAP interpretability figures for an urban-heat XGBoost model.

Simulates a land-surface-temperature (LST) dataset with realistic drivers,
trains an XGBoost regressor, then renders the four canonical SHAP figures:

  shap-sub-bar.jpg         global ranking (mean |SHAP|)
  shap-main-beeswarm.jpg   global summary: importance + direction (the hero)
  shap-sub-dependence.jpg  one feature in depth, interaction-coloured
  shap-sub-waterfall.jpg   local explanation of a single hot neighbourhood
  shap-sub-compose.jpg     bar + dependence side by side (paper-layout trick)
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import xgboost as xgb

OUT = Path(__file__).resolve().parents[2] / "figures"
RNG = np.random.default_rng(20260709)
N = 2400


# ---------------------------------------------------------------- data
def simulate_lst_data(n=N):
    """Urban-heat drivers with realistic signs, interactions and noise."""
    ndvi = np.clip(RNG.beta(2.2, 2.8, n), 0.02, 0.95)            # vegetation
    build = np.clip(RNG.beta(2.5, 2.2, n), 0.02, 0.98)           # building density
    albedo = np.clip(RNG.normal(0.16, 0.05, n), 0.05, 0.40)      # surface albedo
    water_d = RNG.gamma(2.0, 1.6, n)                             # km to water
    pop = np.clip(RNG.lognormal(8.2, 0.9, n) / 1000, 0.05, 60)   # k persons/km2
    road = np.clip(RNG.gamma(2.4, 2.4, n), 0.2, 25)              # road km/km2
    svf = np.clip(RNG.normal(0.62, 0.16, n), 0.15, 1.0)          # sky view factor
    elev = np.clip(RNG.normal(45, 30, n), 0, 160)                # elevation (m)

    lst = (
        34.0
        - 6.5 * ndvi                       # vegetation cools
        + 4.8 * build                      # built-up warms
        - 9.0 * albedo                     # bright surfaces cool
        + 0.35 * np.minimum(water_d, 6)    # further from water = warmer (saturates)
        + 0.9 * np.log1p(pop)              # anthropogenic heat
        + 0.06 * road
        - 1.6 * svf * ndvi                 # open + green interact
        - 0.012 * elev
        + RNG.normal(0, 0.8, n)            # measurement noise
    )
    X = pd.DataFrame({
        "NDVI": ndvi, "Building density": build, "Albedo": albedo,
        "Dist. to water (km)": water_d, "Population (k/km2)": pop,
        "Road density": road, "Sky view factor": svf, "Elevation (m)": elev,
    })
    return X, pd.Series(lst, name="LST (degC)")


X, y = simulate_lst_data()

# ---------------------------------------------------------------- model
model = xgb.XGBRegressor(
    n_estimators=400, max_depth=4, learning_rate=0.06,
    subsample=0.9, colsample_bytree=0.9, random_state=0,
)
model.fit(X, y)

# ---------------------------------------------------------------- SHAP values
explainer = shap.TreeExplainer(model)          # exact + fast for tree models
sv = explainer(X)                              # shap.Explanation (n x features)


def save(name, w=8.2, h=None):
    fig = plt.gcf()
    if h is not None:
        fig.set_size_inches(w, h)
    fig.savefig(OUT / name, dpi=150, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"saved {name}")


# 1) global ranking: mean |SHAP| bar
shap.plots.bar(sv, max_display=8, show=False)
save("shap-sub-bar.jpg", w=7.6, h=4.6)

# 2) the beeswarm: importance + direction + distribution, one panel
shap.plots.beeswarm(sv, max_display=8, show=False)
save("shap-main-beeswarm.jpg", w=8.6, h=5.4)

# 3) dependence: NDVI in depth, coloured by its strongest interaction
shap.plots.scatter(sv[:, "NDVI"], color=sv[:, "Sky view factor"], show=False)
save("shap-sub-dependence.jpg", w=7.6, h=5.2)

# 4) waterfall: why is THIS neighbourhood hot? (highest predicted LST)
hot_i = int(np.argmax(model.predict(X)))
shap.plots.waterfall(sv[hot_i], max_display=9, show=False)
save("shap-sub-waterfall.jpg", w=8.0, h=5.6)

# 5) composing SHAP panels manually (plt.sca trick) for a paper layout
fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.8))
plt.sca(axes[0])
shap.plots.bar(sv, max_display=8, show=False, ax=axes[0])
plt.sca(axes[1])
shap.plots.scatter(sv[:, "NDVI"], color=sv[:, "Sky view factor"],
                   show=False, ax=axes[1])
axes[0].set_title("(a) Global importance", loc="left", fontsize=11)
axes[1].set_title("(b) NDVI dependence", loc="left", fontsize=11)
fig.tight_layout()
fig.savefig(OUT / "shap-sub-compose.jpg", dpi=150, facecolor="white",
            bbox_inches="tight")
plt.close(fig)
print("saved shap-sub-compose.jpg")
print("done")
