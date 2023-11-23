import flwr as fl
from typing import Dict


# Define the global value for the number of clients and the training round
NUM_CLIENTS = 3
ROUNDS = 2


# Return the current round
def fit_config(server_round: int) -> Dict:
    config = {
        "server_round": server_round,
    }
    return config


# Aggregate metrics and calculate weighted averages
def metrics_aggregate(results) -> Dict:
    if not results:
        return {}

    else:
        total_samples = 0  # Number of samples in the dataset

        # Collecting metrics
        aggregated_metrics = {
            "Accuracy": 0,
            "Precision": 0,
            "Recall": 0,
            "F1_Score": 0,
        }

        # Extracting values from the results
        for samples, metrics in results:
            for key, value in metrics.items():
                if key not in aggregated_metrics:
                    aggregated_metrics[key] = 0
                else:
                    aggregated_metrics[key] += (value * samples)
            total_samples += samples

        # Compute the weighted average for each metric
        for key in aggregated_metrics.keys():
            aggregated_metrics[key] = round(aggregated_metrics[key] / total_samples, 6)

        return aggregated_metrics


if __name__ == "__main__":

    print(f"Server:\n")

    # Build a strategy
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=NUM_CLIENTS,
        min_evaluate_clients=NUM_CLIENTS,
        min_available_clients=NUM_CLIENTS,
        on_fit_config_fn=fit_config,
        evaluate_metrics_aggregation_fn=metrics_aggregate,
        fit_metrics_aggregation_fn=metrics_aggregate,
    )

    # Generate a text file for saving the server log
    fl.common.logger.configure(identifier="FL_Test", filename="log.txt")

    # Start the server
    fl.server.start_server(
        config=fl.server.ServerConfig(num_rounds=ROUNDS),
        strategy=strategy,
        server_address="127.0.0.1:8080",
    )
