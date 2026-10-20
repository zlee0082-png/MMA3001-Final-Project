# Provide the tools used by the reusable prediction function.

import numpy as np
import pandas as pd

from pandas.api.types import (
    is_numeric_dtype,
    is_bool_dtype,
    is_complex_dtype
)
# Validate new inputs and predict scaled sound pressure level in dB.
def predict_airfoil(inputs, bundle):
    """
    Predict scaled sound pressure level from airfoil conditions.

    Parameters
    ----------
    inputs : pandas.DataFrame
        One observation per row, containing exactly:
        frequency_hz: frequency in Hz.
        angle_deg: angle of attack in degrees.
        chord_m: chord length in m.
        velocity_m_s: free-stream velocity in m/s.
        displacement_thickness_m: displacement thickness in m.

        Values must be finite real numbers within the recorded
        development ranges. All features except angle must be positive.

    bundle : dict
        Fitted pipeline and metadata loaded from the saved model file.

    Returns
    -------
    pandas.Series
        Predicted scaled SPL in dB, preserving the input row index.

    Raises
    ------
    TypeError
        If inputs are not a DataFrame or contain unsupported data types.
    ValueError
        If inputs are empty, columns are incorrect, values are missing,
        or values violate the supported input limits.
    """
    if not isinstance(inputs, pd.DataFrame):
        raise TypeError("Inputs must be a pandas DataFrame.")

    if inputs.empty:
        raise ValueError("Provide at least one input row.")

    if inputs.columns.has_duplicates:
        raise ValueError("Duplicate column names are not supported.")

    schema = bundle["metadata"]
    required = schema["features"]

    missing = [name for name in required if name not in inputs.columns]
    extra = [name for name in inputs.columns if name not in required]

    if missing or extra:
        raise ValueError(
            f"Column mismatch. Missing: {missing}; extra: {extra}"
        )

    features = inputs.loc[:, required].copy()

    for name in required:
        dtype = features[name].dtype

        if (
            not is_numeric_dtype(dtype)
            or is_bool_dtype(dtype)
            or is_complex_dtype(dtype)
        ):
            raise TypeError(
                f"{name} must contain real numeric values."
            )

    if features.isna().any().any():
        raise ValueError("Missing input values are not supported.")

    features = features.astype(float)

    if not np.isfinite(features.to_numpy()).all():
        raise ValueError("Infinite input values are not supported.")

    if (features[schema["positive_features"]] <= 0).any().any():
        raise ValueError(
            "Frequency, chord, velocity and thickness must be positive."
        )

    lower = pd.Series(schema["input_minimum"])
    upper = pd.Series(schema["input_maximum"])

    outside = features.lt(lower) | features.gt(upper)

    if outside.any().any():
        columns = outside.columns[outside.any()].tolist()

        raise ValueError(
            f"Inputs outside supported development ranges: {columns}"
        )

    predictions = bundle["pipeline"].predict(features)

    return pd.Series(
        predictions,
        index=inputs.index,
        name=schema["target"]
    )