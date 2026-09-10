import pandas as pd
import matplotlib.pyplot as plt
from app.ml_service.vision_core import DistanceEstimator


def main():

    df = pd.read_csv("app/data/distance_dataset.csv")

    print("Dataset:")
    print(df)

    pixels_width = df["pixel_width"].tolist()

    ground_truth = df["ground_truth_distance"].tolist()

    print("\nPixel widths:")
    print(pixels_width)

    print("\nGround truth:")
    print(ground_truth)


    estimator = DistanceEstimator(
        focal_length=800,
        real_car_width_m=1.8
    )

    # Prediction
    predictions = []

    for width in pixels_width:

        distance = estimator.calculate_distance(width)

        predictions.append(distance)

    print("Predictions:", predictions)

     # Add prediction to DataFrame
    df["prediction"] = predictions

    # Error
    df["error"] = (
        df["prediction"]
        - df["ground_truth_distance"]
    )

    # Absolute error
    df["absolute_error"] = df["error"].abs()

    print(
    df.gr`oupby("vehicle_type")["absolute_error"].mean()
)

    df["percentage_error"] = (
    df["absolute_error"]
    / df["ground_truth_distance"]
    * 100
)
    mape = df["percentage_error"].mean()

    print(f"MAPE: {mape:.2f}%")

    bad_predictions = df[
        df["percentage_error"] > 10
    ]

    print("\nBad predictions:")
    print(bad_predictions)

    print("\nEvaluation result:")
    print(df)

    # MAE
    mae = df["absolute_error"].mean()

    print(f"\nMAE: {mae:.2f}m")

    # RMSE
    rmse = (
        (df["error"] ** 2).mean()
    ) ** 0.5

    print(f"RMSE: {rmse:.2f}m")

    plt.figure(figsize=(8,6))

    # plt.scatter(
    #     df["ground_truth_distance"],
    #     df["prediction"]
    # )

    # plt.plot(
    #     df["ground_truth_distance"],
    #     df["ground_truth_distance"],
    #     linestyle="--"
    # )

    # plt.xlabel("Ground Truth Distance (m)")
    # plt.ylabel("Predicted Distance (m)")
    # plt.title("Ground Truth vs Prediction")
    plt.scatter(
    df["ground_truth_distance"],
    df["error"]
    )

    # Đường error = 0
    plt.axhline(
        y=0,
        linestyle="--"
    )

    plt.xlabel("Ground Truth Distance (m)")
    plt.ylabel("Prediction Error (m)")
    plt.title("Prediction Error vs Ground Truth")


    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    main()