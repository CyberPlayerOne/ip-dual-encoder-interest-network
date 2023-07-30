SHELL := /bin/bash

pipreqsnb:
	pipreqsnb .

mlflow:
	mlflow server --host 0.0.0.0 --backend-store-uri /home/ubuntu/mlruns
mlflow_local:
	mlflow server --host 0.0.0.0 --backend-store-uri ./mlruns --port 5001

tensorboard:
	tensorboard --host 0.0.0.0 --logdir /home/ubuntu/tbruns
tensorboard_local:
	tensorboard --host 0.0.0.0 --logdir ./tbruns --port 6007