def main():

    ground_truth = [20, 30, 40, 50, 60]

    predictions = [21, 28, 3, 47, 65]

    print("Ground Truth:", ground_truth)
    print("Predictions:", predictions)

    errors = []

    for gt, pred in zip(ground_truth, predictions):
        error = pred - gt
        errors.append(error)

    print("Errors:", errors)

    absolute_errors = []

    for error in errors:
        absolute_error = abs(error)
        absolute_errors.append(absolute_error)

    print("Absolute Errors:", absolute_errors)

    mae = sum(absolute_errors) / len(absolute_errors)

    print(f"MAE: {mae:.2f}m")

    squared_errors = []

    for error in errors:
        squared_error = error ** 2
        squared_errors.append(squared_error)

    mse = sum(squared_errors) / len(squared_errors)

    rmse = mse ** 0.5

    print(f"RMSE: {rmse:.2f}m")


if __name__ == "__main__":
    main()