# flake8: noqa: E501
#
# En este dataset se desea pronosticar el default (pago) del cliente el próximo
# mes a partir de 23 variables explicativas.
#
#   LIMIT_BAL: Monto del credito otorgado. Incluye el credito individual y el
#              credito familiar (suplementario).
#         SEX: Genero (1=male; 2=female).
#   EDUCATION: Educacion (0=N/A; 1=graduate school; 2=university; 3=high school; 4=others).
#    MARRIAGE: Estado civil (0=N/A; 1=married; 2=single; 3=others).
#         AGE: Edad (years).
#       PAY_0: Historia de pagos pasados. Estado del pago en septiembre, 2005.
#       PAY_2: Historia de pagos pasados. Estado del pago en agosto, 2005.
#       PAY_3: Historia de pagos pasados. Estado del pago en julio, 2005.
#       PAY_4: Historia de pagos pasados. Estado del pago en junio, 2005.
#       PAY_5: Historia de pagos pasados. Estado del pago en mayo, 2005.
#       PAY_6: Historia de pagos pasados. Estado del pago en abril, 2005.
#   BILL_AMT1: Historia de pagos pasados. Monto a pagar en septiembre, 2005.
#   BILL_AMT2: Historia de pagos pasados. Monto a pagar en agosto, 2005.
#   BILL_AMT3: Historia de pagos pasados. Monto a pagar en julio, 2005.
#   BILL_AMT4: Historia de pagos pasados. Monto a pagar en junio, 2005.
#   BILL_AMT5: Historia de pagos pasados. Monto a pagar en mayo, 2005.
#   BILL_AMT6: Historia de pagos pasados. Monto a pagar en abril, 2005.
#    PAY_AMT1: Historia de pagos pasados. Monto pagado en septiembre, 2005.
#    PAY_AMT2: Historia de pagos pasados. Monto pagado en agosto, 2005.
#    PAY_AMT3: Historia de pagos pasados. Monto pagado en julio, 2005.
#    PAY_AMT4: Historia de pagos pasados. Monto pagado en junio, 2005.
#    PAY_AMT5: Historia de pagos pasados. Monto pagado en mayo, 2005.
#    PAY_AMT6: Historia de pagos pasados. Monto pagado en abril, 2005.
#
# La variable "default payment next month" corresponde a la variable objetivo.
#
# El dataset ya se encuentra dividido en conjuntos de entrenamiento y prueba
# en la carpeta "files/input/".
#
# Los pasos que debe seguir para la construcción de un modelo de
# clasificación están descritos a continuación.
#
#
# Paso 1.
# Realice la limpieza de los datasets:
# - Renombre la columna "default payment next month" a "default".
# - Remueva la columna "ID".
# - Elimine los registros con informacion no disponible.
# - Para la columna EDUCATION, valores > 4 indican niveles superiores
#   de educación, agrupe estos valores en la categoría "others".
#
# Renombre la columna "default payment next month" a "default"
# y remueva la columna "ID".
#
#
# Paso 2.
# Divida los datasets en x_train, y_train, x_test, y_test.
#
#
# Paso 3.
# Cree un pipeline para el modelo de clasificación. Este pipeline debe
# contener las siguientes capas:
# - Transforma las variables categoricas usando el método
#   one-hot-encoding.
# - Escala las demas variables al intervalo [0, 1].
# - Selecciona las K mejores caracteristicas.
# - Ajusta un modelo de regresion logistica.
#
#
# Paso 4.
# Optimice los hiperparametros del pipeline usando validación cruzada.
# Use 10 splits para la validación cruzada. Use la función de precision
# balanceada para medir la precisión del modelo.
#
#
# Paso 5.
# Guarde el modelo (comprimido con gzip) como "files/models/model.pkl.gz".
# Recuerde que es posible guardar el modelo comprimido usanzo la libreria gzip.
#
#
# Paso 6.
# Calcule las metricas de precision, precision balanceada, recall,
# y f1-score para los conjuntos de entrenamiento y prueba.
# Guardelas en el archivo files/output/metrics.json. Cada fila
# del archivo es un diccionario con las metricas de un modelo.
# Este diccionario tiene un campo para indicar si es el conjunto
# de entrenamiento o prueba. Por ejemplo:
#
# {'type': 'metrics', 'dataset': 'train', 'precision': 0.8, 'balanced_accuracy': 0.7, 'recall': 0.9, 'f1_score': 0.85}
# {'type': 'metrics', 'dataset': 'test', 'precision': 0.7, 'balanced_accuracy': 0.6, 'recall': 0.8, 'f1_score': 0.75}
#
#
# Paso 7.
# Calcule las matrices de confusion para los conjuntos de entrenamiento y
# prueba. Guardelas en el archivo files/output/metrics.json. Cada fila
# del archivo es un diccionario con las metricas de un modelo.
# de entrenamiento o prueba. Por ejemplo:
#
# {'type': 'cm_matrix', 'dataset': 'train', 'true_0': {"predicted_0": 15562, "predicte_1": 666}, 'true_1': {"predicted_0": 3333, "predicted_1": 1444}}
# {'type': 'cm_matrix', 'dataset': 'test', 'true_0': {"predicted_0": 15562, "predicte_1": 650}, 'true_1': {"predicted_0": 2490, "predicted_1": 1420}}
#
import gzip
import json
import os
import pickle

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder


def load_data():
    train = pd.read_csv("files/input/train_data.csv.zip")
    test = pd.read_csv("files/input/test_data.csv.zip")

    train.rename(
        columns={"default payment next month": "default"},
        inplace=True,
    )
    test.rename(
        columns={"default payment next month": "default"},
        inplace=True,
    )

    train.drop(columns=["ID"], inplace=True)
    test.drop(columns=["ID"], inplace=True)

    train = train[
        (train["EDUCATION"] != 0)
        & (train["MARRIAGE"] != 0)
    ].copy()

    test = test[
        (test["EDUCATION"] != 0)
        & (test["MARRIAGE"] != 0)
    ].copy()

    train["EDUCATION"] = train["EDUCATION"].apply(
        lambda x: 4 if x > 4 else x
    )
    test["EDUCATION"] = test["EDUCATION"].apply(
        lambda x: 4 if x > 4 else x
    )

    return train, test


def create_pipeline(x_train):
    categorical_features = [
        "SEX",
        "EDUCATION",
        "MARRIAGE",
    ]

    numerical_features = [
        column
        for column in x_train.columns
        if column not in categorical_features
    ]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                categorical_features,
            ),
            (
                "numerical",
                MinMaxScaler(),
                numerical_features,
            ),
        ]
    )

    pipeline = Pipeline(
        [
            ("preprocessor", preprocessor),
            ("selectkbest", SelectKBest(score_func=f_classif)),
            (
                "logisticregression",
                LogisticRegression(
                    max_iter=1000,
                    random_state=12345,
                ),
            ),
        ]
    )

    param_grid = {
        "selectkbest__k": list(range(1, 24)),
        "logisticregression__C": [
            0.01,
            0.1,
            1,
            10,
            100,
        ],
        "logisticregression__solver": [
            "lbfgs",
            "saga",
        ],
    }

    model = GridSearchCV(
        pipeline,
        param_grid=param_grid,
        cv=10,
        scoring="balanced_accuracy",
        n_jobs=-1,
    )

    return model


def save_model(model):
    os.makedirs("files/models", exist_ok=True)

    with gzip.open(
        "files/models/model.pkl.gz",
        "wb",
    ) as file:
        pickle.dump(model, file)


def metrics_record(dataset, y_true, y_pred):
    return {
        "type": "metrics",
        "dataset": dataset,
        "precision": round(
            precision_score(y_true, y_pred),
            3,
        ),
        "balanced_accuracy": round(
            balanced_accuracy_score(y_true, y_pred),
            3,
        ),
        "recall": round(
            recall_score(y_true, y_pred),
            3,
        ),
        "f1_score": round(
            f1_score(y_true, y_pred),
            3,
        ),
    }


def cm_record(dataset, y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)

    return {
        "type": "cm_matrix",
        "dataset": dataset,
        "true_0": {
            "predicted_0": int(cm[0, 0]),
            "predicted_1": int(cm[0, 1]),
        },
        "true_1": {
            "predicted_0": int(cm[1, 0]),
            "predicted_1": int(cm[1, 1]),
        },
    }


def save_metrics(metrics):
    os.makedirs("files/output", exist_ok=True)

    with open(
        "files/output/metrics.json",
        "w",
        encoding="utf-8",
    ) as file:
        for metric in metrics:
            file.write(json.dumps(metric) + "\n")


def main():
    train, test = load_data()

    x_train = train.drop(columns=["default"])
    y_train = train["default"]

    x_test = test.drop(columns=["default"])
    y_test = test["default"]

    model = create_pipeline(x_train)
    model.fit(x_train, y_train)

    save_model(model)

    y_train_pred = model.predict(x_train)
    y_test_pred = model.predict(x_test)

    metrics = [
        metrics_record(
            "train",
            y_train,
            y_train_pred,
        ),
        metrics_record(
            "test",
            y_test,
            y_test_pred,
        ),
        cm_record(
            "train",
            y_train,
            y_train_pred,
        ),
        cm_record(
            "test",
            y_test,
            y_test_pred,
        ),
    ]

    save_metrics(metrics)


if __name__ == "__main__":
    main()