# Import libraries
import mlflow
import mlflow.tensorflow

# Log experiment
def mlflow_experiment(params):
    """
    Autolog experiment into MLflow and Databricks.
    :param params: Parameters to be logged
    """
    # Set experiment in databricks
    mlflow.set_tracking_uri('databricks')
    mlflow.set_experiment('/Shared/autonomous vehicle')

    # Enable TensorFlow autologging
    mlflow.tensorflow.autolog()

    # Start run
    mlflow.start_run()
    
    # Log parameters
    mlflow.log_params(params)