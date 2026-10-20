# Read airfoil conditions from a CSV and save model predictions.

import argparse
from pathlib import Path

import joblib
import pandas as pd

from airfoil_prediction import predict_airfoil


def main():
    """
    Run prediction from command-line file paths.

    The input CSV must contain the five named features in their
    original units. The output CSV contains those features and
    predicted_scaled_spl_db.

    Invalid inputs or file-access errors terminate the command
    with an error message and a nonzero exit status.
    """
    parser = argparse.ArgumentParser(
        description="Predict airfoil scaled sound pressure level."
    )

    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="CSV containing the five input features."
    )

    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Destination CSV for the predictions."
    )

    parser.add_argument(
        "--model",
        type=Path,
        default=Path("models") / "final_airfoil_model.joblib",
        help="Path to your saved model bundle."
    )

    args = parser.parse_args()

    try:
        # Prevent accidentally replacing the input or model file.
        if args.output.resolve() in {
            args.input.resolve(),
            args.model.resolve()
        }:
            raise ValueError(
                "The output path must differ from the input and model paths."
            )

        # Load the fitted pipeline and read the original-unit inputs.
        bundle = joblib.load(args.model)

        inputs = pd.read_csv(
            args.input,
            float_precision="round_trip"
        )

        # Apply the shared input checks and fitted prediction pipeline.
        predictions = predict_airfoil(inputs, bundle)

        result = inputs.loc[
            :, bundle["metadata"]["features"]
        ].copy()

        result["predicted_scaled_spl_db"] = predictions

        # Create the destination folder and save one result per input row.
        args.output.parent.mkdir(parents=True, exist_ok=True)

        result.to_csv(
            args.output,
            index=False
        )

    except (OSError, ValueError, TypeError) as error:
        parser.exit(
            status=1,
            message=f"Prediction failed: {error}\n"
        )

    print(f"Predicted {len(result)} rows.")
    print(f"Saved predictions to: {args.output}")


if __name__ == "__main__":
    main()