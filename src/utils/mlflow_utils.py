import mlflow

def start_run(experiment_name, run_name):
    mlflow.set_experiment(experiment_name = experiment_name)
    mlflow.start_run(run_name = run_name)

def log_params(params: dict):
    mlflow.log_params(params)

def log_metrics(metrics: dict):
    mlflow.log_metrics(metrics)