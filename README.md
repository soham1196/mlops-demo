# MlOPS Demo

# Documentation

## Code structure

| File/folder                   | Description                                |
| ----------------------------- | ------------------------------------------ |
| `run.py`                        | End to End MlOps demonstration using Mlflow. It includes model train, model version registration, model validation, model deployment |
| `eda.py`                  | Exploratory Data Analysis of the Olist Dataset . |
| `utils.py`         | Adhoc utility functions. |

## Data - https://www.kaggle.com/olistbr/marketing-funnel-olist

It contains two datasets:
- olist_closed_deals_dataset.csv 
- olist_market_qualified_leads.csv

**Concept**: Join the two dataset using mql_id. If the lead has a seller we can say the lead is converted.

**Problem Statement** : Create a model to determine whether a lead is converted or not.

## MLFLOW

MLflow is an open source platform for managing the end-to-end machine learning lifecycle. It tackles four primary functions:

- Tracking experiments to record and compare parameters and results (MLflow Tracking).

- Packaging ML code in a reusable, reproducible form in order to share with other data scientists or transfer to production (MLflow Projects).

- Managing and deploying models from a variety of ML libraries to a variety of model serving and inference platforms (MLflow Models).

- Providing a central model store to collaboratively manage the full lifecycle of an MLflow Model, including model versioning, stage transitions, and annotations (MLflow Model Registry).

- MLflow is library-agnostic. You can use it with any machine learning library, and in any programming language, since all functions are accessible through a REST API and CLI. For convenience, the project also includes a Python API, R API, and Java API.


