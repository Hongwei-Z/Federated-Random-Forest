import flwr as fl
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings("ignore")


def fit_round(server_round: int):
    return {"server_round": server_round}


def metrics_aggregate(results):
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

        # Compute the average for each metric
        for key in aggregated_metrics.keys():
            aggregated_metrics[key] /= total_samples

        return aggregated_metrics


if __name__ == "__main__":

    model = RandomForestClassifier(
        n_estimators=100,
        class_weight='balanced',
        criterion='entropy',
    )

    strategy = fl.server.strategy.FedAvg(
        min_available_clients=2,
        on_fit_config_fn=fit_round,
        evaluate_metrics_aggregation_fn=metrics_aggregate,
        fit_metrics_aggregation_fn=metrics_aggregate,
    )

    fl.common.logger.configure(identifier="FL_Test", filename="log_server.txt")

    fl.server.start_server(
        server_address="127.0.0.1:8080",
        strategy=strategy,
        config=fl.server.ServerConfig(num_rounds=1),
    )
