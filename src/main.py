from constants import *
from sample import Sample
from decisionTreeClassifier import DecisionTreeClassifier
from decisionTreeRegressor import DecisionTreeRegressor
from randomFlorestClassifier import RandomFlorestClassifier
from randomFlorestRegressor import RandomFlorestRegressor
from mlp import MLP, MinMaxScalerSym, classe_via_regressao

from typing import Literal
import random
import numpy as np

def buildSample(line: str, modelType: Literal["classifier", "regressor"]) -> Sample:
    lineList = line.strip().split(",")

    id = int(lineList[0])

    # ignores: id (0), inrelevant data (1 and 2) and outputs (6 and 7)
    attributesStr = lineList[3:6]

    attributesFloat = [float(attr) for attr in attributesStr]

    if modelType == "regressor":
        output = float(lineList[6])
    elif modelType == "classifier":
        output = int(lineList[7])

    return Sample(id, attributesFloat, output)

def trainingSet(modelType: Literal["classifier", "regressor"]) -> tuple[list[Sample], set[int]]:
    with open(DATA_FILE_PATH, "r") as f:
        lines = f.readlines()

    numLines = len(lines)

    trainingSetSize = int(numLines * TRAINING_SET_PERCENT)

    chosenIndexes = set(
        random.sample(range(numLines), trainingSetSize)
    )

    samples = []
    for i, line in enumerate(lines):
        if i in chosenIndexes:
            samples.append(buildSample(line, modelType))

    return samples, chosenIndexes

# choosenIndexes contain the file indexes/lines used to create the training set
def testSet(choosenIndexes: set[int], modelType: Literal["classifier", "regressor"]) -> list[Sample]:
    with open(DATA_FILE_PATH, "r") as f:
        samples = []
        for i, line in enumerate(f):
            if i not in choosenIndexes:
                samples.append(buildSample(line, modelType))

    return samples



# MLP helpers: convertem Sample para arrays numpy esperados pela classe MLP


def samplesToArrays(samples: list[Sample]) -> tuple[np.ndarray, np.ndarray]:
    """Converte lista de Sample para arrays (X, y).

    X tem shape (n, 3) com [qPA, pulso, resp].
    y tem shape (n,) com a saída (gravidade contínua ou classe inteira).
    """
    X = np.array([s.attributesList for s in samples], dtype=float)
    y = np.array([s.output for s in samples], dtype=float)
    return X, y


def addEngineeredFeatures(X_raw: np.ndarray) -> np.ndarray:
    """Adiciona features derivadas |pulso-80| e qPA² às 3 originais.

    Sai de (n, 3) → (n, 5). Justificativa vem da exploração dos dados, pulso atua
    por desvio do normal (≈80 bpm) e qPA tem efeito quadrático/direcional
    """
    qPA = X_raw[:, 0]
    pulso = X_raw[:, 1]
    return np.column_stack([
        X_raw,
        np.abs(pulso - 80.0),
        qPA ** 2,
    ])


def trainMLPRegressor(trainingSamples: list[Sample]) -> tuple[MLP, MinMaxScalerSym, MinMaxScalerSym]:
    """Treina o MLP regressor com a melhor versão que encontramos na nossa modelagem

    Arquitetura [5, 8, 5, 3, 1]:
        - 3 camadas ocultas em funil estreito (8 → 5 → 3 neurônios)
        - 1 saída linear (gravidade contínua)
        - 115 parâmetros — escolhida via ablação de capacidade

    Hiperparâmetros (lr=0.05, momentum=0.9, l2_lambda=0.0) vêm dos
    estudos de momentum e regularização do notebook de desenvolvimento.

    Returns:
        (mlp_treinado, scaler_X, scaler_y) — os scalers são necessários para
        inverter a normalização da gravidade prevista nas predições futuras.
    """
    # Converte amostras em arrays
    X_raw, y_train = samplesToArrays(trainingSamples)
    X_train = addEngineeredFeatures(X_raw)              # 3D → 5D
    y_train = y_train.reshape(-1, 1)

    # Normaliza min-max para [-1, +1] — ajusta scalers apenas no treino
    scaler_X = MinMaxScalerSym().fit(X_train)
    scaler_y = MinMaxScalerSym().fit(y_train)
    X_train_n = scaler_X.transform(X_train)
    y_train_n = scaler_y.transform(y_train)

    # Treina a MLP
    mlp = MLP(
        layer_sizes=[5, 8, 5, 3, 1],
        task="regression",
        lr=0.05,
        momentum=0.9,
        seed=42,
        l2_lambda=0.0,                                   # Apesar de implementar a L2, não mostrou ganhos substanciais
    )
    mlp.fit(
        X_train_n, y_train_n,
        epochs=400, batch_size=32, patience=30,
        verbose=False,
    )

    return mlp, scaler_X, scaler_y


def testMLPRegressor(mlp: MLP, scaler_X: MinMaxScalerSym, scaler_y: MinMaxScalerSym,
                     testSamples: list[Sample]) -> float:
    """Avalia o MLP regressor em amostras de teste e retorna o MSE

    Espelha a interface de DecisionTreeRegressor.testTree() para
    consistência na comparação entre modelos.
    """
    X_raw, y_true = samplesToArrays(testSamples)
    X_test = addEngineeredFeatures(X_raw)
    X_test_n = scaler_X.transform(X_test)

    # Predição desnormaliza para escala original da gravidade
    y_pred_n = mlp.predict(X_test_n)
    y_pred = scaler_y.inverse_transform(y_pred_n).flatten()

    mse = float(np.mean((y_true - y_pred) ** 2))
    return mse


if __name__ == "__main__":
    trainingSamples, chosenIndexes = trainingSet("regressor")
    testSamples = testSet(chosenIndexes, "regressor")

    # Decision Tree Regressor
    tree = DecisionTreeRegressor()
    tree.buildTree(trainingSamples)
    print("Decision Tree Regressor MSE:", tree.testTree(testSamples))

    # Random Forest Regressor
    florest = RandomFlorestRegressor()
    florest.buildFlorest(trainingSamples)
    print("Random Forest Regressor   MSE:", florest.testFlorest(testSamples))

    # MLP Regressor — Iter. 6 do desenvolvimento (ver notebook)
    mlp, scaler_X, scaler_y = trainMLPRegressor(trainingSamples)
    print("MLP Regressor             MSE:", testMLPRegressor(mlp, scaler_X, scaler_y, testSamples))
