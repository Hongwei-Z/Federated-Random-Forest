import helper
import numpy as np
import flwr as fl
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import log_loss, accuracy_score, precision_score, recall_score, f1_score
import warnings
warnings.simplefilter('ignore')


# Create the flower client
class FlowerClient(fl.client.NumPyClient):

    # Get the current local model parameters
    def get_parameters(self, config):
        print(f"Client {client_id} received the parameters.")
        return helper.get_params(model)

    # Train the local model, return the model parameters to the server
    def fit(self, parameters, config):
        print("Parameters before setting: ", parameters)
        helper.set_params(model, parameters)
        print("Parameters after setting: ", model.get_params())

        model.fit(X_train, y_train)
        print(f"Training finished for round {config['server_round']}.")

        trained_params = helper.get_params(model)
        print("Trained Parameters: ", trained_params)

        return trained_params, len(X_train), {}

    # Evaluate the local model, return the evaluation result to the server
    def evaluate(self, parameters, config):
        helper.set_params(model, parameters)

        y_pred = model.predict(X_test)
        loss = log_loss(y_test, y_pred, labels=[0, 1])

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')

        line = "-" * 21
        print(line)
        print(f"Accuracy : {accuracy:.8f}")
        print(f"Precision: {precision:.8f}")
        print(f"Recall   : {recall:.8f}")
        print(f"F1 Score : {f1:.8f}")
        print(line)

        return loss, len(X_test), {"Accuracy": accuracy, "Precision": precision, "Recall": recall, "F1_Score": f1}


if __name__ == "__main__":
    client_id = 1
    print(f"Client {client_id}:\n")

    # Get the dataset for local model
    X_train, y_train, X_test, y_test = helper.load_dataset(client_id - 1)

    # Print the label distribution
    unique, counts = np.unique(y_train, return_counts=True)
    train_counts = dict(zip(unique, counts))
    print("Label distribution in the training set:", train_counts)
    unique, counts = np.unique(y_test, return_counts=True)
    test_counts = dict(zip(unique, counts))
    print("Label distribution in the testing set:", test_counts, '\n')

    # Create and fit the local model
    model = RandomForestClassifier(
        class_weight='balanced',
        criterion='entropy',
        n_estimators=100,
        max_depth=40,
        min_samples_split=2,
        min_samples_leaf=1,
    )
    model.fit(X_train, y_train)

    # Start the client
    fl.client.start_numpy_client(server_address="127.0.0.1:8080", client=FlowerClient())
