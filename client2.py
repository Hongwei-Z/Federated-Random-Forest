import helper
import flwr as fl
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import log_loss, accuracy_score, precision_score, recall_score, f1_score
import warnings
warnings.filterwarnings("ignore")


class FlowerClient(fl.client.NumPyClient):

    def get_parameters(self, config):
        return helper.get_model_params(model)

    def fit(self, parameters, config):
        print("Parameters before setting: ", parameters)
        helper.set_model_params(model, parameters)
        print("Parameters after setting: ", model.get_params())

        model.fit(X_train, y_train)
        print(f"Training finished for round {config['server_round']}.")

        trained_params = helper.get_model_params(model)
        print("Trained Parameters: ", trained_params)

        return trained_params, len(X_train), {}

    def evaluate(self, parameters, config):
        helper.set_model_params(model, parameters)

        y_pred_proba = model.predict_proba(X_test)
        loss = log_loss(y_test, y_pred_proba, labels=[0, 1])

        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')

        return loss, len(X_test), {"Accuracy": accuracy, "Precision": precision, "Recall": recall, "F1_Score": f1}


if __name__ == "__main__":
    client_id = 1
    X_train, y_train, X_test, y_test = helper.load_dataset(client_id)

    model = RandomForestClassifier(
        n_estimators=100,
        class_weight='balanced',
        criterion='entropy',
    )
    model.fit(X_train, y_train)

    fl.client.start_numpy_client(server_address="127.0.0.1:8080", client=FlowerClient())
