# Glossary

## Well geometry

| Term | Definition |
|---|---|
| **MD** | Measured Depth (ft) — total cable/drill length from surface |
| **TVD** | True Vertical Depth (ft) — vertical distance below surface (= Z column) |
| **TVT** | True Vertical Thickness (ft) — vertical thickness of the formation at the wellbore; the **prediction target** |
| **TST** | True Stratigraphic Thickness - bed-normal thickness; useful distinction when beds dip and vertical thickness is not the same as thickness perpendicular to bedding |
| **TVT_input** | Copy of TVT provided as a feature; contains NaN in the evaluation zone |
| **Lateral** | Horizontal portion of the wellbore; this is where TVT is to be predicted |
| **Kick-off point** | Depth where the well transitions from vertical to horizontal |
| **Evaluation zone** | Portion of the lateral where TVT_input is NaN and must be predicted |
| **X, Y** | Easting / Northing coordinates in feet |
| **Z** | True Vertical Depth in feet (negative = below surface) |

## Logs & geology

| Term | Definition |
|---|---|
| **GR** | Gamma Ray log (API units) — measures natural radioactivity; high GR = shale, low GR = sand/carbonate |
| **Typewell** | Vertical reference well used for geological correlation; provides GR vs. TVT profile |
| **Formation** | Named rock unit (e.g., EGFDL, BUDA, ANCC) |
| **Formation top** | Depth or stratigraphic position where a named formation begins; direct top distances are train-only oracle features unless inferred from test-safe evidence |
| **Stratigraphic state** | Latent formation-relative position of a row or well segment; includes likely zone/top proximity and uncertainty |
| **Log motif** | Repeating pattern in a well log sequence, such as a GR peak/trough shape, used for correlation rather than as a single row value |
| **DTW** | Dynamic Time Warping - sequence-alignment method that can match similar log motifs under lag or local stretching |
| **Geosteering** | Real-time or retrospective interpretation of well trajectory relative to stratigraphy, often using logs and structural context |
| **ANCC** | Formation name (predicted depth column in horizontal well) |
| **ASTNU / ASTNL** | Formation name upper/lower |
| **EGFDU / EGFDL** | Formation name upper/lower |
| **BUDA** | Formation name |
| **Cross-section** | 2D slice through the subsurface showing well path + geology (visualized in .png files) |

## Modeling

| Term | Definition |
|---|---|
| **RMSE** | Root Mean Squared Error — competition evaluation metric |
| **GP** | Gaussian Process — non-parametric Bayesian model for spatial/sequential data |
| **BNN** | Bayesian Neural Network — neural network with probabilistic weights |
| **SVI** | Stochastic Variational Inference — scalable approximate Bayesian inference |
| **HMC/NUTS** | Hamiltonian Monte Carlo / No-U-Turn Sampler — exact MCMC for Bayesian posteriors |
| **NumpPyro** | JAX-backed probabilistic programming library |
| **Flax** | JAX neural network library |
| **Optax** | JAX gradient optimizers library |

## Repository

| Term | Definition |
|---|---|
| **Run bundle** | Folder under `artifacts/logs/runs/<run_id>/` containing all metadata for one experiment |
| **Artifact** | Any generated output (figure, table, model, report) — never hand-edited |
| **Run ID** | `YYYY-MM-DD__HHMM__<tag>__<git_sha>` — unique experiment identifier |
