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

   1. **`helper.py`**:  
    Path of the dataset in the `load_dataset()` method.
      ```python
      df = pd.read_csv('data.csv')
      ```
      If you want to add more clients, you need to modify the `load_dataset()` method so that the dataset is split into more sub-datasets.
   2. **`server.py`**:  
    Modify the global values so that it meets your requirements.
      ```python
      NUM_CLIENTS = 3
      ROUNDS = 2
      ```
   3. **`client1.py, client2.py, client3.py`**:  
    Duplicate more client files if you need to, change the Client ID in the `main()`.
      ```python
      client_id = 1  # Client 1
      ```
   4. (Optional) **`server.py, client(1,2,3).py`**:   
    If the run process results in a server address error, try the default address. Modify `server_address` in the server file and the client files respectively.
      ```python
      server_address="0.0.0.0:8080"
      ```
   
    


### Running

1. **Install Required Libraries**:
   ```bash
   pip install -r requirements.txt
   ```
2. **Running the Code**:

    You have two options to run the code:

   - **Option 1: Using `run.bat`**:

     Double-click on `run.bat` file. This will automatically execute the necessary scripts.

   - **Option 2: Using Command Line**:

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



Author: Hongwei Zhang

Citation:   
1. [Flower Framework Documentation](https://flower.dev/docs/framework/index.html)   
2. [scikit-learn RandomForestClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)