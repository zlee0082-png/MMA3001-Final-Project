#MMA3001-Final-Project
# Airfoil Self-Noise Prediction

**MMA3001: Numerical Methods and Machine Learning**

This project predicts **scaled sound pressure level (SPL), in decibels, at a specified frequency** using five airfoil and flow variables.

The workflow compares regression models, tunes model parameters, investigates preprocessing, evaluates reserved test observations, and provides a reusable command-line prediction tool.

The selected model is a **Radial Basis Function Support Vector Regressor (RBF SVR)** with logarithmic preprocessing and standardisation.

Recorded performance:

- Mean grouped cross-validation RMSE: **2.829 dB**
- Final-test RMSE: **2.410 dB**
- Mean-baseline final-test RMSE: **6.853 dB**

## Licence

The project's original source code is licensed under the MIT License.
See [LICENSE](LICENSE).

The UCI Airfoil Self-Noise dataset is separately licensed under
CC BY 4.0 and retains its original attribution and licence.

## Source Code Documentation

HTML API documentation is stored in `docs/`:

- `docs/airfoil_prediction.html`: prediction function, input requirements,
  output and error handling.
- `docs/predict.html`: command-line prediction workflow.

To view the documentation, download or clone the repository and open
the HTML files in a web browser.

To regenerate the documentation after editing the source code:

```bash
python -m pip install pdoc
python -m pdoc --docformat numpy -o docs airfoil_prediction.py predict.py
```
## Automated Tests

The tests check the saved prediction interface, invalid-input handling,
column ordering, output row alignment and command-line CSV prediction.

Run from the repository's main folder using the project Python environment:

```bash
python -m pip install pytest pytest-html
python -m pytest tests -v --html=results/test_report.html --self-contained-html
```

The tests require:

- `models/final_airfoil_model.joblib`
- `examples/prediction_inputs.csv`

Open `results/test_report.html` in a browser to inspect the results.

These tests verify software behaviour. Predictive accuracy is assessed
separately through grouped cross-validation and reserved-test evaluation.

## 1. Engineering Problem

Airfoil self-noise affects the acoustic performance of aerodynamic components. Measuring noise experimentally for every combination of operating conditions requires time and resources.

This project develops an empirical model that estimates noise from measured airfoil and flow conditions. The intended use is preliminary investigation within the domain represented by the dataset.

The output is scaled SPL at a specified frequency. It does not directly represent total noise across the frequency spectrum, sound power, lift or drag.

## 2. Dataset

The project uses the **Airfoil Self-Noise dataset** from the UCI Machine Learning Repository.

- Observations: **1,503**
- Input variables: **5**
- Target variable: **scaled sound pressure level**
- Original filename: `airfoil_self_noise.dat`
- Format: whitespace-separated numerical data without a header

UCI describes the dataset as aerodynamic and acoustic measurements involving NACA 0012 airfoils under different wind-tunnel conditions.

Dataset source:

https://archive.ics.uci.edu/dataset/291/airfoil+self+noise

The raw dataset should be located at:
data/raw/airfoil_self_noise.dat

If it is missing, download and extract the dataset from UCI and place the `.dat` file at this location. Record its source, download date and any modifications in `data/README.md`.

## 3. Inputs and Outputs

### Model inputs

| Column name | Meaning | Unit |
| --- | --- | --- |
| `frequency_hz` | Frequency at which noise is estimated | Hz |
| `angle_deg` | Angle of attack | degrees |
| `chord_m` | Airfoil chord length | m |
| `velocity_m_s` | Free-stream velocity | m/s |
| `displacement_thickness_m` | Suction-side boundary-layer displacement thickness | m |

Provide inputs in their **original units**. The saved pipeline performs the required logarithmic transformation and standardisation automatically.

### Measured target

The dataset column `scaled_spl_db` contains the measured scaled SPL in dB. It is used for model training and evaluation.

### Prediction output

The command-line prediction file contains the five input columns and:

| Output column | Meaning | Unit |
| --- | --- | --- |
| `predicted_scaled_spl_db` | Estimated scaled SPL for the supplied conditions and frequency | dB |

One prediction is produced for each input row.

### Supported input domain

The prediction function requires:

- Exactly the five named input columns.
- At least one input row.
- Finite real numerical values.
- Strictly positive frequency, chord, velocity and displacement thickness.
- Values within the inclusive development-data minimum and maximum for every feature.

Angle may equal zero.

The saved limits are recorded in:
models/final_airfoil_metadata.json

Individual range checks do not establish that every accepted combination is physically realistic or well represented in the training data. The code also cannot identify every plausible mistake in the supplied units.

## 4. Repository Contents

| File or location | Purpose |
| --- | --- |
| `README.md` | Project overview and instructions |
| Main analysis notebook (`.ipynb`) | Complete analysis, model development and verification workflow |
| `airfoil_prediction.py` | Shared prediction function and input validation |
| `predict.py` | Command-line prediction interface |
| `data/raw/airfoil_self_noise.dat` | Original numerical dataset |
| `data/README.md` | Dataset provenance and modification record |
| `models/final_airfoil_model.joblib` | Saved fitted pipeline and metadata |
| `models/final_airfoil_metadata.json` | Input requirements and software versions |
| `examples/prediction_inputs.csv` | Example inputs exported from development observations |
| `examples/new_conditions.csv` | Conditions entered for a new prediction |
| `results/` | Result tables, figures, predictions and verification records |
| Project report | Detailed engineering discussion, validation, optimisation and AI reflection |

The model files and most result files are generated by running the notebook.

## 5. Software and Setup

The project uses Python with:

- NumPy
- pandas
- Matplotlib
- SciPy
- scikit-learn
- joblib
- threadpoolctl
- ipykernel

VS Code's Python and Jupyter extensions are used to run the notebook.

The notebook uses `GroupKFold(shuffle=True, random_state=42)`, which requires **scikit-learn 1.6 or later**.

### Create an environment on Windows

Open the repository folder in VS Code. Run these commands in its terminal.

If you already have a working project environment, continue using it. Otherwise:

```powershell
python -m venv .venv
```

If the repository contains the recorded dependency file:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

If `requirements.txt` has not yet been created, install the required packages:

```powershell
.\.venv\Scripts\python.exe -m pip install numpy pandas matplotlib scipy "scikit-learn>=1.6,<2" joblib threadpoolctl ipykernel
```

This fallback creates a new environment. Train and export the model in that environment before using it.

Select `.venv` as the Python kernel for the notebook. Calling the environment's Python executable directly avoids needing to activate it in PowerShell.

After verifying a complete run, record the working package versions:

```powershell
.\.venv\Scripts\python.exe -m pip freeze | Out-File -FilePath requirements.txt -Encoding utf8
```

For an existing saved model, use the Python and package versions recorded in its metadata. Loading a model with different scikit-learn versions is not supported.

## 6. Reproducing the Analysis

Run the notebook from the repository root so relative paths such as `data/raw/` and `models/` resolve correctly.

Restart the notebook kernel and run the cells in order.

| Stage | Main activities |
| --- | --- |
| Data preparation | Load the dataset, define inputs and target, and audit data quality |
| Grouped splitting | Separate development and final-test groups and create validation folds |
| Baseline comparison | Compare mean, linear, Ridge and SVR models |
| Hyperparameter tuning | Search Ridge and RBF SVR settings |
| Preprocessing investigation | Compare standardisation with logarithmic preprocessing followed by standardisation |
| Model selection and testing | Fix the model choice, fit on development data and evaluate reserved test observations |
| Sensitivity analysis | Investigate input perturbations and changes to a training target |
| Performance analysis | Benchmark fitting and prediction, compare batch processing and profile function calls |
| Model export | Save the fitted pipeline and metadata |
| Verification | Check saved-model predictions, column ordering, row alignment and invalid-input handling |

Keep `airfoil_prediction.py` and `predict.py` in the repository root.

Rerunning notebook cells can overwrite their saved output files.

## 7. Computational Method

### Baseline models

Five initial models were compared:

1. Mean predictor.
2. Ordinary linear regression.
3. Ridge regression.
4. SVR with a linear kernel.
5. SVR with an RBF kernel.

The mean predictor establishes a reference without input-dependent predictions. Linear models test whether an additive linear relationship is sufficient. RBF SVR allows nonlinear relationships between the inputs and SPL.

### Preprocessing

For the selected model:

1. Apply `log10` to frequency, chord, velocity and displacement thickness.
2. Leave angle unlogged.
3. Standardise the resulting five inputs.
4. Fit the RBF SVR.

The target remains in dB.

Preprocessing and the estimator are combined in a scikit-learn pipeline. Scaling parameters are learned only from the training observations within each validation fold.

### Hyperparameter search

| Parameter | Values tested |
| --- | --- |
| Ridge `alpha` | 0.01, 0.1, 1, 10, 100 |
| SVR `C` | 1, 10, 100 |
| SVR `gamma` | `"scale"`, 0.05, 0.5 |
| SVR `epsilon` | 0.1, 0.5 |

The SVR grid contains 18 parameter combinations.

The selected configuration was:

```text
Candidate: SVR_log_scaled
Kernel: RBF
C: 100
gamma: 0.05
epsilon: 0.5 dB
```

## 8. Validation Design

Rows sharing the same combination of angle, chord and velocity are assigned to the same condition group.

The dataset was divided using:

```python
GroupShuffleSplit(
    n_splits=1,
    test_size=0.20,
    random_state=42,
)
```

The recorded split contains:

- **1,182 development observations**
- **321 final-test observations**

The 20% fraction refers to condition groups, so the row counts do not have to follow an exact 80/20 ratio.

Five-fold grouped cross-validation is applied within the development data. Each group is used for validation once, and all candidates use the same folds.

The selection criterion is the **lowest mean validation RMSE across the five folds**. The final-test observations are reserved for evaluating the fixed model.

Additional metrics include:

- **MAE:** mean absolute prediction error, in dB.
- **R²:** predictive performance relative to the evaluated targets' mean.
- **Mean residual:** average measured minus predicted SPL, in dB.

A positive residual indicates underprediction.

The recorded group-overlap checks found no shared groups between development and final-test data or between training and validation within a fold.

## 9. Recorded Results

### Model comparison

| Configuration | Mean grouped CV RMSE (dB) |
| --- | ---: |
| Mean predictor | 6.929 |
| Ordinary linear regression | 4.728 |
| Initial scaled Ridge | 4.728 |
| Initial scaled linear SVR | 4.797 |
| Initial scaled RBF SVR | 4.035 |
| Tuned scaled Ridge | 4.727 |
| Tuned log-scaled Ridge | 5.217 |
| Tuned scaled RBF SVR | 3.477 |
| Selected log-scaled RBF SVR | **2.829** |

The selected SVR reduced validation RMSE by **29.9%** relative to the initial RBF SVR.

Both preprocessing configurations were tuned. Their difference therefore compares the resulting tuned configurations rather than isolating the effect of logarithms at fixed parameters.

### Reserved final-test performance

| Model | Final-test RMSE (dB) |
| --- | ---: |
| Selected log-scaled RBF SVR | **2.410** |
| Mean baseline | 6.853 |

### Prediction performance

For the recorded benchmark using 200 observations:

| Workflow | Median time (ms) |
| --- | ---: |
| One observation per prediction call | 696.893 |
| Complete batch in one prediction call | 17.955 |

Batch prediction was **38.81 times faster**.

The recorded maximum prediction difference was **0.0 dB**, checked using an absolute tolerance of `1e-10` dB.

Runtime depends on the machine, software versions and workload. These values describe the recorded project run.

## 10. Predicting Noise for New Conditions

### Step 1: Ensure the model has been saved

Run the notebook through the model-export stage.

The following file must exist:

```text
models/final_airfoil_model.joblib
```

`predict.py` loads this fitted model. It does not train a new model.

### Step 2: Create the input CSV

Create or edit:

```text
examples/new_conditions.csv
```

Use these exact column names:

```csv
frequency_hz,angle_deg,chord_m,velocity_m_s,displacement_thickness_m
200,21.0,0.02541,35,0.03
```

Each additional row represents another set of conditions.

Save the CSV before running the prediction command. The example above was accepted by the recorded model; the metadata determines the limits for any retrained model.

### Step 3: Run prediction

From the repository root, using the working project environment:

```powershell
python predict.py --input examples/new_conditions.csv --output results/new_noise_predictions.csv
```

Alternatively, call the virtual environment directly:

```powershell
.\.venv\Scripts\python.exe predict.py --input examples/new_conditions.csv --output results/new_noise_predictions.csv
```

The default model location is:

```text
models/final_airfoil_model.joblib
```

A different model bundle can be specified using `--model`.

The output path must differ from the input and model paths. An existing output CSV at the chosen destination is overwritten.

### Step 4: Read the predictions

Open:

```text
results/new_noise_predictions.csv
```

The recorded example produced:

```csv
frequency_hz,angle_deg,chord_m,velocity_m_s,displacement_thickness_m,predicted_scaled_spl_db
200,21.0,0.02541,35,0.03,117.50626719978455
```

This corresponds to approximately **117.51 dB scaled SPL at 200 Hz** for the supplied conditions.

This is a model estimate. No measured ground truth was supplied for this example.

## 11. Using the Prediction Function in Python

```python
import joblib
import pandas as pd

from airfoil_prediction import predict_airfoil

# Load the fitted pipeline and its input requirements.
bundle = joblib.load("models/final_airfoil_model.joblib")

# Read inputs in their original units.
inputs = pd.read_csv(
    "examples/prediction_inputs.csv",
    float_precision="round_trip",
)

# Validate the inputs and predict scaled SPL.
predictions = predict_airfoil(inputs, bundle)

print(predictions)
```

`predict_airfoil` returns a pandas Series named `scaled_spl_db`, preserving the input row index.

The command-line script writes the predictions under the column name `predicted_scaled_spl_db`.

Only load model bundles from trusted sources.

## 12. Data Quality and Failure Handling

Before modelling, the notebook checks:

- Missing or non-numeric entries across all six columns.
- Positive or negative infinity.
- Nonpositive values in frequency, chord, velocity and displacement thickness.
- Exact duplicate rows.

The audit is saved to:

```text
results/data_audit.csv
```

Invalid numerical or physical-sign entries cause an error. Duplicate rows are counted and reported. The audit does not automatically remove or replace observations.

The prediction function rejects:

- Unsupported input containers.
- Empty inputs.
- Missing, extra or duplicate columns.
- Text, Boolean or complex DataFrame dtypes.
- Missing or infinite values.
- Nonpositive values in the four positive-only inputs.
- Values outside the saved development ranges.

CSV numerical fields are parsed by pandas before the shared function checks them.

For handled input or file errors, the command-line script prints an explanatory message and exits with a nonzero status.

## 13. Verification and Saved Evidence

The notebook verifies:

- Separation of condition groups.
- Agreement between original and reloaded model predictions.
- Equivalent predictions after input columns are reordered.
- Preservation of input row indices.
- Rejection of common invalid inputs.
- Agreement between individual and batch predictions.

Important result files include:

| File in `results/` | Contents |
| --- | --- |
| `data_audit.csv` | Dataset quality checks |
| `split_manifest.csv` | Development/test allocation |
| `group_split_checks.csv` | Group-overlap checks |
| `baseline_comparison.csv` | Initial model comparison |
| `model_selection_comparison.csv` | Tuned candidate comparison |
| `model_selection.json` | Selected model and parameters |
| `final_test_metrics.csv` | Reserved-test performance |
| `final_test_predictions.csv` | Individual test predictions and errors |
| `final_test_diagnostics.png` | Prediction and residual plots |
| `input_sensitivity_results.csv` | Input perturbation results |
| `training_observation_sensitivity_summary.csv` | Training-target perturbation results |
| `runtime_summary.csv` | Model timing comparison |
| `batch_timing_summary.csv` | Individual versus batch prediction timings |
| `benchmark_details.json` | Benchmark settings and environment |
| `prediction_verification.csv` | Invalid-input verification results |

## 14. Troubleshooting

| Problem | What to check |
| --- | --- |
| Dataset not found | Confirm the working directory and `data/raw/airfoil_self_noise.dat` |
| Saved model not found | Complete the notebook's export stage |
| `No module named airfoil_prediction` | Keep `airfoil_prediction.py` in the repository root and check the notebook working directory |
| `GroupKFold` rejects `shuffle` | Confirm the selected kernel uses scikit-learn 1.6 or later |
| Input outside supported ranges | Compare the saved CSV with `final_airfoil_metadata.json` |
| Predictions appear to use old inputs | Save the CSV and inspect the file at the path passed to `--input` |
| Model loading fails in another environment | Match the recorded software versions or retrain and export in the new environment |

For example, chord `0.01` m is below the recorded model's minimum chord of `0.0254` m and is rejected.

## 15. Limitations

- Performance for other airfoil profiles, experiments or operating environments has not been established.
- Final testing uses held-out groups from the same dataset rather than an independent experiment.
- Condition groups are approximated from angle, chord and velocity.
- The winning CV score was used for model selection and may be optimistic.
- The selected `C=100` was the upper boundary of the tested search grid.
- Accepted feature ranges do not guarantee reliable predictions for every combination.
- Displacement thickness must be supplied from a measurement or estimate.
- No calibrated prediction intervals or application-specific acceptable error threshold were established.
- The SVR epsilon band is a training-loss setting, not an uncertainty bound.
- Sensitivity results apply only to the disturbances tested.
- Peak memory and scalability across larger training datasets were not measured.

## 16. Documentation and AI Use

The main notebook contains explanations and function docstrings. The accompanying project report discusses the engineering context, method selection, validation, optimisation, limitations and lessons learned.

ChatGPT/Codex assisted with workflow planning, substantial initial code and report drafts, explanations and debugging. Numerical results were obtained by executing the project workflow.

The report's AI reflection records how suggestions were checked and which decisions remained the student's responsibility.

## 17. Version Control

GitHub is used to maintain the code and documentation during development.

Meaningful commits should record changes to the notebook, Python files, documentation, dependency information and relevant results.

Keep the local virtual environment and temporary files out of version control. The saved model should either be included or be reproducible by following the notebook instructions.

## 18. References and Data Licence

Brooks, T., Pope, D., & Marcolini, M. (1989). *Airfoil Self-Noise* [Dataset]. UCI Machine Learning Repository.

https://doi.org/10.24432/C5VW2C

The source dataset is licensed under **Creative Commons Attribution 4.0 International (CC BY 4.0)**:

https://creativecommons.org/licenses/by/4.0/

The dataset licence does not automatically determine the licence for the project code.

Software documentation:

- [scikit-learn GroupKFold](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GroupKFold.html)
- [scikit-learn SVR](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html)
- [scikit-learn model persistence](https://scikit-learn.org/stable/model_persistence.html)