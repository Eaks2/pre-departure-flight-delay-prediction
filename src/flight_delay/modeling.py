"""Training-only preprocessing and chronological model selection.

Spark ML references:
https://spark.apache.org/docs/latest/ml-features.html
https://spark.apache.org/docs/latest/ml-classification-regression.html#logistic-regression
"""

from __future__ import annotations

import re
from typing import Dict, List, Sequence, Tuple

from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.classification import LogisticRegression, LogisticRegressionModel
from pyspark.ml.feature import (
    Imputer,
    OneHotEncoder,
    StandardScaler,
    StringIndexer,
    VectorAssembler,
)
from pyspark.sql import DataFrame

from .evaluation import threshold_metric_rows


def build_preprocessor(
    categorical_features: Sequence[str],
    numeric_features: Sequence[str],
) -> Pipeline:
    indexed_columns = [f"{name}__index" for name in categorical_features]
    encoded_columns = [f"{name}__one_hot" for name in categorical_features]
    imputed_columns = [f"{name}__imputed" for name in numeric_features]

    indexers = [
        StringIndexer(
            inputCol=input_column,
            outputCol=output_column,
            handleInvalid="keep",
            stringOrderType="frequencyDesc",
        )
        for input_column, output_column in zip(categorical_features, indexed_columns)
    ]
    encoder = OneHotEncoder(
        inputCols=indexed_columns,
        outputCols=encoded_columns,
        dropLast=True,
        handleInvalid="keep",
    )
    imputer = Imputer(
        inputCols=list(numeric_features),
        outputCols=imputed_columns,
        strategy="median",
    )
    numeric_assembler = VectorAssembler(
        inputCols=imputed_columns,
        outputCol="numeric_features_unscaled",
        handleInvalid="error",
    )
    scaler = StandardScaler(
        inputCol="numeric_features_unscaled",
        outputCol="numeric_features_scaled",
        withMean=False,
        withStd=True,
    )
    final_assembler = VectorAssembler(
        inputCols=[*encoded_columns, "numeric_features_scaled"],
        outputCol="features",
        handleInvalid="error",
    )
    return Pipeline(stages=[*indexers, encoder, imputer, numeric_assembler, scaler, final_assembler])


def fit_preprocessor(
    training: DataFrame,
    categorical_features: Sequence[str],
    numeric_features: Sequence[str],
) -> PipelineModel:
    return build_preprocessor(categorical_features, numeric_features).fit(training)


def _selection_key(row: dict) -> tuple:
    return (
        row["f1"],
        row["precision"],
        row["recall"],
        row["roc_auc"],
        -abs(row["threshold"] - 0.5),
        -row["reg_param"],
    )


def select_logistic_model(
    training: DataFrame,
    validation: DataFrame,
    regularization_grid: Sequence[float],
    threshold_grid: Sequence[float],
    max_iterations: int,
) -> Tuple[LogisticRegressionModel, dict, List[dict]]:
    """Tune L2 regularization and threshold on validation F1 only."""

    all_rows: List[dict] = []
    models: Dict[float, LogisticRegressionModel] = {}
    for reg_param in regularization_grid:
        estimator = LogisticRegression(
            featuresCol="features",
            labelCol="label",
            predictionCol="prediction",
            probabilityCol="probability",
            rawPredictionCol="rawPrediction",
            family="binomial",
            regParam=float(reg_param),
            elasticNetParam=0.0,
            maxIter=int(max_iterations),
            standardization=False,
        )
        model = estimator.fit(training)
        models[float(reg_param)] = model
        validation_predictions = model.transform(validation).cache()
        common = {
            "model": "logistic_regression",
            "split": "validation",
            "reg_param": float(reg_param),
            "elastic_net_param": 0.0,
            "max_iterations": int(max_iterations),
        }
        all_rows.extend(
            threshold_metric_rows(validation_predictions, threshold_grid, common)
        )
        validation_predictions.unpersist()

    best_row = max(all_rows, key=_selection_key)
    best_model = models[float(best_row["reg_param"])]
    return best_model, dict(best_row), all_rows


def fit_final_logistic(
    training: DataFrame,
    reg_param: float,
    max_iterations: int,
) -> LogisticRegressionModel:
    estimator = LogisticRegression(
        featuresCol="features",
        labelCol="label",
        predictionCol="prediction",
        probabilityCol="probability",
        rawPredictionCol="rawPrediction",
        family="binomial",
        regParam=float(reg_param),
        elasticNetParam=0.0,
        maxIter=int(max_iterations),
        standardization=False,
    )
    return estimator.fit(training)


def feature_names_from_metadata(transformed: DataFrame) -> List[str]:
    metadata = transformed.schema["features"].metadata.get("ml_attr", {})
    attribute_groups = metadata.get("attrs", {})
    number_of_attributes = int(metadata.get("num_attrs", 0))
    indexed_names = {}
    for attributes in attribute_groups.values():
        for attribute in attributes:
            indexed_names[int(attribute["idx"])] = attribute.get(
                "name", f"feature_{attribute['idx']}"
            )
    if number_of_attributes <= 0 and indexed_names:
        number_of_attributes = max(indexed_names) + 1
    return [indexed_names.get(index, f"feature_{index}") for index in range(number_of_attributes)]


def coefficient_rows(
    model: LogisticRegressionModel,
    transformed: DataFrame,
    numeric_feature_names: Sequence[str] = (),
) -> List[dict]:
    names = feature_names_from_metadata(transformed)
    values = list(model.coefficients.toArray())
    if len(names) != len(values):
        names = [f"feature_{index}" for index in range(len(values))]
    if numeric_feature_names:
        renamed = []
        for name in names:
            match = re.fullmatch(r"numeric_features_scaled_(\d+)", name)
            numeric_index = int(match.group(1)) if match else -1
            if 0 <= numeric_index < len(numeric_feature_names):
                renamed.append(f"numeric__{numeric_feature_names[numeric_index]}")
            else:
                renamed.append(name)
        names = renamed
    rows = []
    for index, (name, coefficient) in enumerate(zip(names, values)):
        coefficient = float(coefficient)
        rows.append(
            {
                "feature_index": index,
                "feature_name": name,
                "coefficient": coefficient,
                "absolute_coefficient": abs(coefficient),
                "direction": (
                    "higher_delay_risk"
                    if coefficient > 0
                    else "lower_delay_risk"
                    if coefficient < 0
                    else "neutral"
                ),
            }
        )
    rows.sort(key=lambda row: row["absolute_coefficient"], reverse=True)
    return rows
