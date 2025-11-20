#!/bin/bash
mkdir -p mlflow_artifacts
mlflow ui \
    --backend-store-uri sqlite:///mlflow_artifacts/mlruns.db \
    --default-artifact-root file:./mlflow_artifacts/artifacts \
    --host 0.0.0.0 \
    --port 5000
