# Federated Random Forest
## Using [Flower](https://flower.dev/docs/framework/index.html) federated learning with sklearn [Random Forest Classifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html) 
This project implements a Federated Random Forest (FRF) using the federated learning library Flower and the sklearn random forest classifier. It is demonstrated through three clients as an example.


### Getting Started
1. This project contains following source code files:
    ```
    helper.py
    server.py
    client1.py
    client2.py
    client3.py
    run.bat
    ```

2. Changes to be made in the code before you start:

   1. **`helper.py`:**    
    Path of the dataset in the `load_dataset()` method.
      ```python
      df = pd.read_csv('data.csv')
      ```
      If you want to add more clients, you need to modify the `load_dataset()` method so that the dataset is split into more sub-datasets.
   2. **`server.py`:**    
    Modify the global values so that it meets your requirements.
      ```python
      NUM_CLIENTS = 3
      ROUNDS = 2
      ```
   3. **`client1.py, client2.py, client3.py`:**  
    Duplicate more client files if you need to, change the Client ID in the `main()`.
      ```python
      client_id = 1  # Client 1
      ```
   4. (Optional) **`server.py, client(1,2,3).py`:**   
    If the run process results in a server address error, try the default address. Modify `server_address` in the server file and the client files respectively.
      ```python
      server_address="0.0.0.0:8080"
      ```
   
    


### Running

1. **Install Required Libraries:**
   ```bash
   pip install -r requirements.txt
   ```
2. **Running the Code:**

    You have two options to run the code:

   - **Option 1: Using `run.bat`:**

     Double-click on `run.bat` file. This will automatically execute the necessary scripts.

   - **Option 2: Using Command Line:**

     If you prefer to run the scripts manually, execute the following commands in your terminal:

     1. Loading the helper `helper.py`:
        ```bash
        python helper.py
        ```
     2. Start the server `server.py`:
        ```bash
        python server.py
        ```
     3. Run three clients in three terminals:
        ```bash
        python client1.py
        ```
        ```bash
        python client2.py
        ```
        ```bash
        python client3.py
        ```




### Evaluating

1. **Example of running results on the server:**
   ```bash
    Server:

    INFO flwr 2023-11-23 17:18:58,487 | app.py:162 | Starting Flower server, config: ServerConfig(num_rounds=2, round_timeout=None)
    INFO flwr 2023-11-23 17:18:58,487 | app.py:175 | Flower ECE: gRPC server running (2 rounds), SSL is disabled
    INFO flwr 2023-11-23 17:18:58,487 | server.py:89 | Initializing global parameters
    INFO flwr 2023-11-23 17:18:58,487 | server.py:276 | Requesting initial parameters from one random client
    INFO flwr 2023-11-23 17:19:40,389 | server.py:280 | Received initial parameters from one random client
    INFO flwr 2023-11-23 17:19:40,389 | server.py:91 | Evaluating initial parameters
    INFO flwr 2023-11-23 17:19:40,389 | server.py:104 | FL starting
    DEBUG flwr 2023-11-23 17:19:41,214 | server.py:222 | fit_round 1: strategy sampled 3 clients (out of 3)
    DEBUG flwr 2023-11-23 17:20:22,909 | server.py:236 | fit_round 1 received 3 results and 0 failures
    DEBUG flwr 2023-11-23 17:20:22,909 | server.py:173 | evaluate_round 1: strategy sampled 3 clients (out of 3)
    DEBUG flwr 2023-11-23 17:20:23,613 | server.py:187 | evaluate_round 1 received 3 results and 0 failures
    DEBUG flwr 2023-11-23 17:20:23,613 | server.py:222 | fit_round 2: strategy sampled 3 clients (out of 3)
    DEBUG flwr 2023-11-23 17:21:04,206 | server.py:236 | fit_round 2 received 3 results and 0 failures
    DEBUG flwr 2023-11-23 17:21:04,206 | server.py:173 | evaluate_round 2: strategy sampled 3 clients (out of 3)
    DEBUG flwr 2023-11-23 17:21:04,914 | server.py:187 | evaluate_round 2 received 3 results and 0 failures
    INFO flwr 2023-11-23 17:21:04,914 | server.py:153 | FL finished in 84.52697419992182
    INFO flwr 2023-11-23 17:21:04,914 | app.py:225 | app_fit: losses_distributed [(1, 2.2396674950917563), (2, 2.224978526433309)]
    INFO flwr 2023-11-23 17:21:04,914 | app.py:226 | app_fit: metrics_distributed_fit {'Accuracy': [(1, 0.0), (2, 0.0)], 'Precision': [(1, 0.0), (2, 0.0)], 'Recall': [(1, 0.0), (2, 0.0)], 'F1_Score': [(1, 0.0), (2, 0.0)]}
    INFO flwr 2023-11-23 17:21:04,914 | app.py:227 | app_fit: metrics_distributed {'Accuracy': [(1, 0.937862), (2, 0.93827)], 'Precision': [(1, 0.937341), (2, 0.93773)], 'Recall': [(1, 0.937862), (2, 0.93827)], 'F1_Score': [(1, 0.932165), (2, 0.932682)]}
    INFO flwr 2023-11-23 17:21:04,914 | app.py:228 | app_fit: losses_centralized []
    INFO flwr 2023-11-23 17:21:04,914 | app.py:229 | app_fit: metrics_centralized {}
   ```
2. **Example of running results on the client:**
    ```bash
    Client 1:

    Label distribution in the training set: {0: 149362, 1: 24035}
    Label distribution in the testing set: {0: 37175, 1: 6175}
    
    INFO flwr 2023-11-23 17:19:40,377 | grpc.py:49 | Opened insecure gRPC connection (no certificates were passed)
    DEBUG flwr 2023-11-23 17:19:40,389 | connection.py:42 | ChannelConnectivity.IDLE
    DEBUG flwr 2023-11-23 17:19:40,389 | connection.py:42 | ChannelConnectivity.CONNECTING
    DEBUG flwr 2023-11-23 17:19:40,389 | connection.py:42 | ChannelConnectivity.READY
    Client 1 received the parameters.
    Parameters before setting:  [array(100), array(40), array(2), array(1)]
    Parameters after setting:  {'bootstrap': True, 'ccp_alpha': 0.0, 'class_weight': 'balanced', 'criterion': 'entropy', 'max_depth': 40, 'max_features': 'sqrt', 'max_leaf_nodes': None, 'max_samples': None, 'min_impurity_decrease': 0.0, 'min_samples_leaf': 1, 'min_samples_split': 2, 'min_weight_fraction_leaf': 0.0, 'n_estimators': 100, 'n_jobs': None, 'oob_score': False, 'random_state': None, 'verbose': 0, 'warm_start': False}
    Training finished for round 1.
    Trained Parameters:  [100, 40, 2, 1]
    ---------------------
    Accuracy : 0.93718570
    Precision: 0.93695380
    Recall   : 0.93718570
    F1 Score : 0.93131784
    ---------------------
    Parameters before setting:  [array(100), array(40), array(2), array(1)]
    Parameters after setting:  {'bootstrap': True, 'ccp_alpha': 0.0, 'class_weight': 'balanced', 'criterion': 'entropy', 'max_depth': 40, 'max_features': 'sqrt', 'max_leaf_nodes': None, 'max_samples': None, 'min_impurity_decrease': 0.0, 'min_samples_leaf': 1, 'min_samples_split': 2, 'min_weight_fraction_leaf': 0.0, 'n_estimators': 100, 'n_jobs': None, 'oob_score': False, 'random_state': None, 'verbose': 0, 'warm_start': False}
    Training finished for round 2.
    Trained Parameters:  [100, 40, 2, 1]
    ---------------------
    Accuracy : 0.93757785
    Precision: 0.93744875
    Recall   : 0.93757785
    F1 Score : 0.93174330
    ---------------------
    DEBUG flwr 2023-11-23 17:21:04,932 | connection.py:139 | gRPC channel closed
    INFO flwr 2023-11-23 17:21:04,932 | app.py:215 | Disconnect and shut down
    ```



Author: Hongwei Zhang

Citation:   
1. [Flower Framework Documentation](https://flower.dev/docs/framework/index.html)   
2. [scikit-learn RandomForestClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)