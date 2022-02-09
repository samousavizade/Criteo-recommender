#!/bin/bash
docker build --tag mlflow:latest ./MLFlow
docker build --tag preprocessing:latest ./Preprocessing
docker-compose up
