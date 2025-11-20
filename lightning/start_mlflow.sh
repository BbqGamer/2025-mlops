#!/bin/bash
mlflow ui \
    --backend-store-uri sqlite:///mlruns.db \
    --host 0.0.0.0 \
    --port 5000
