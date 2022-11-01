import tensorflow as tf
import mlflow
from Config import mlflow_config

def parser_mlflow():
    return dict({
        'experiment_name': mlflow_config.experiment_name,
        'run_name_model_type': mlflow_config.run_name_model_type,
        'model_name': mlflow_config.model_name,
        'mlflow_autolog': mlflow_config.mlflow_autolog,
        'tensorflow_autolog': mlflow_config.tensorflow_autolog,
        'mlflow_custom_log': mlflow_config.mlflow_custom_log,
        'user_name': mlflow_config.user_name,
    })

def mllog_run():
    args_mlflow = parser_mlflow()
    print("Options-Mlflow:")
    for k, v in args_mlflow.items():
        print(f"  {k}: {v}")

    args_mlflow['model_name'] = None if not args_mlflow['model_name'] or args_mlflow['model_name'] == "None" else \
        args_mlflow['model_name']

    if args_mlflow['tensorflow_autolog']:
        mlflow.tensorflow.autolog()
    if args_mlflow['mlflow_autolog']:
        mlflow.autolog()

    try:
        exp_id = mlflow.create_experiment(name=args_mlflow['experiment_name'])
    except Exception as e:
        exp_id = mlflow.get_experiment_by_name(name=args_mlflow['experiment_name']).experiment_id

    ml_run = mlflow.start_run(experiment_id=exp_id, run_name=args_mlflow['model_name'])
    print("MLflow:")
    print("  run_id:", ml_run.info.run_id)
    print("  experiment_id:", ml_run.info.experiment_id)
    mlflow.set_tag("version.mlflow", mlflow.__version__)
    mlflow.set_tag("version.tensorflow", tf.__version__)
    mlflow.set_tag("mlflow_autolog", args_mlflow['mlflow_autolog'])
    mlflow.set_tag("tensorflow_autolog", args_mlflow['tensorflow_autolog'])
    mlflow.set_tag("mlflow_custom_log", args_mlflow['mlflow_custom_log'])
    mlflow.set_tag("Type of model", args_mlflow['model_name'])
    mlflow.set_tag("Developer", args_mlflow['user_name'])
    return ml_run