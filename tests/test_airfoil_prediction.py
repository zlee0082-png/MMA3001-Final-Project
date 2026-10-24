"""Verify the saved airfoil prediction interface and command-line workflow."""

from pathlib import Path
import subprocess
import sys

import joblib
import numpy as np
import pandas as pd
import pytest

from airfoil_prediction import predict_airfoil


# Locate project files relative to this test file.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "final_airfoil_model.joblib"
EXAMPLE_PATH = PROJECT_ROOT / "examples" / "prediction_inputs.csv"


@pytest.fixture(scope="session")
def bundle():
    """Load the project's saved model once for this test session."""
    return joblib.load(MODEL_PATH)


@pytest.fixture
def valid_inputs(bundle):
    """Read example observations and assign a recognisable row index."""
    inputs = pd.read_csv(
        EXAMPLE_PATH,
        float_precision="round_trip",
    )

    required = bundle["metadata"]["features"]
    inputs = inputs.loc[:, required].head(5).copy()

    # A non-default index checks whether prediction preserves row labels.
    inputs.index = pd.Index(
        range(101, 101 + len(inputs)),
        name="observation",
    )

    return inputs


def test_valid_predictions(bundle, valid_inputs):
    """Check output format, finite values, row alignment and pipeline agreement."""
    predictions = predict_airfoil(valid_inputs, bundle)

    expected = bundle["pipeline"].predict(valid_inputs)

    assert isinstance(predictions, pd.Series)
    assert len(predictions) == len(valid_inputs)
    assert predictions.index.equals(valid_inputs.index)
    assert predictions.name == bundle["metadata"]["target"]
    assert np.isfinite(predictions.to_numpy()).all()

    # The validation wrapper must preserve the fitted pipeline's predictions.
    np.testing.assert_allclose(
        predictions.to_numpy(),
        expected,
        rtol=0,
        atol=1e-10,
    )


def test_column_order_does_not_change_predictions(bundle, valid_inputs):
    """Check that rearranging input columns preserves the predictions."""
    original = predict_airfoil(valid_inputs, bundle)

    reordered = predict_airfoil(
        valid_inputs.loc[:, valid_inputs.columns[::-1]],
        bundle,
    )

    pd.testing.assert_series_equal(
        original,
        reordered,
        check_exact=False,
        rtol=0,
        atol=1e-10,
    )


# Run the invalid-input test separately for each listed case.
@pytest.mark.parametrize(
    "case_name, expected_error",
    [
        ("wrong_container", TypeError),
        ("empty_input", ValueError),
        ("missing_column", ValueError),
        ("extra_column", ValueError),
        ("duplicate_column", ValueError),
        ("text_value", TypeError),
        ("boolean_value", TypeError),
        ("complex_value", TypeError),
        ("missing_value", ValueError),
        ("infinite_value", ValueError),
        ("zero_frequency", ValueError),
        ("outside_range", ValueError),
    ],
)
def test_invalid_inputs_are_rejected(
    bundle,
    valid_inputs,
    case_name,
    expected_error,
):
    """Check that each unsupported input raises the required error type."""
    maximum_velocity = bundle["metadata"]["input_maximum"]["velocity_m_s"]

    invalid_cases = {
        "wrong_container": valid_inputs.to_numpy(),
        "empty_input": valid_inputs.iloc[:0],
        "missing_column": valid_inputs.drop(columns=["frequency_hz"]),
        "extra_column": valid_inputs.assign(unexpected_column=1),
        "duplicate_column": pd.concat(
            [valid_inputs, valid_inputs[["frequency_hz"]]],
            axis=1,
        ),
        "text_value": valid_inputs.assign(frequency_hz="invalid"),
        "boolean_value": valid_inputs.assign(frequency_hz=True),
        "complex_value": valid_inputs.assign(frequency_hz=200 + 1j),
        "missing_value": valid_inputs.assign(frequency_hz=np.nan),
        "infinite_value": valid_inputs.assign(frequency_hz=np.inf),
        "zero_frequency": valid_inputs.assign(frequency_hz=0),
        "outside_range": valid_inputs.assign(
            velocity_m_s=maximum_velocity + 1,
        ),
    }

    # An expected rejection counts as a passing test.
    with pytest.raises(expected_error):
        predict_airfoil(invalid_cases[case_name], bundle)


def test_command_line_prediction(bundle, valid_inputs, tmp_path):
    """Run predict.py and verify the output CSV against direct prediction."""
    # pytest supplies a temporary folder, protecting existing result files.
    input_path = tmp_path / "inputs.csv"
    output_path = tmp_path / "predictions.csv"

    valid_inputs.to_csv(input_path, index=False)

    completed = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "predict.py"),
            "--input",
            str(input_path),
            "--output",
            str(output_path),
            "--model",
            str(MODEL_PATH),
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert output_path.is_file()

    result = pd.read_csv(
        output_path,
        float_precision="round_trip",
    )

    required = bundle["metadata"]["features"]

    assert result.columns.tolist() == required + [
        "predicted_scaled_spl_db"
    ]
    assert len(result) == len(valid_inputs)

    expected = predict_airfoil(valid_inputs, bundle)

    np.testing.assert_allclose(
        result["predicted_scaled_spl_db"].to_numpy(),
        expected.to_numpy(),
        rtol=0,
        atol=1e-10,
    )