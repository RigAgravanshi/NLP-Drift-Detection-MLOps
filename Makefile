setup:
	pip install -r requirements.txt
pull-data:
	python -m src.data.load_data
preprocess:
	python -m src.data.preprocess
simulate:
	python -m src.monitoring.simulate_drift
drift:
	python -m src.monitoring.drift_detector
serve:
	uvicorn src.api.main:app --reload
train:
	python -m src.models.train
evaluate:
	python -m src.models.evaluate
mlflow-ui:
	mlflow ui --backend-store-uri sqlite:///mlflow.db