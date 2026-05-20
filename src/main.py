from constants import *
from sample import Sample
from decisionTreeClassifier import DecisionTreeClassifier
from decisionTreeRegressor import DecisionTreeRegressor
from randomFlorestClassifier import RandomFlorestClassifier
from randomFlorestRegressor import RandomFlorestRegressor
from mlp import MLP, MinMaxScalerSym, classe_via_regressao

from typing import Literal
from sklearn.model_selection import StratifiedKFold
import numpy as np



# Carregamento e construção das amostras

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


def loadAllSamples(modelType: Literal["classifier", "regressor"]) -> list[Sample]:
    """Carrega todas as amostras do dataset com feature engineering aplicada

    O k-fold cross-validation faz seus próprios splits a partir do conjunto
    completo. A feature engineering é aplicada UMA VEZ aqui como passo de
    pré-processamento, garantindo que todos os modelos (árvore, floresta, MLP)
    recebam exatamente as mesmas features 5D — comparação justa.

    Features finais por amostra, transforma o espaço do problema em 5 dimensões ao inves de 3:
        [qPA, pulso, resp, |pulso-80|, qPA²]

    Justificativa vem da exploração dos dados: pulso atua por desvio do
    normal (≈80 bpm) e qPA tem efeito quadrático/direcional na gravidade.
    """
    with open(DATA_FILE_PATH, "r") as f:
        samples = [buildSample(line, modelType) for line in f]

    # Feature engineering uniforme
    for s in samples:
        qPA, pulso, resp = s.attributesList
        s.attributesList = [qPA, pulso, resp, abs(pulso - 80.0), qPA ** 2]

    return samples


# Helpers do MLP — converte Sample para arrays numpy

def samplesToArrays(samples: list[Sample]) -> tuple[np.ndarray, np.ndarray]:
    """Converte lista de Sample para arrays (X, y).

    X tem shape (n, 5) já com feature engineering aplicada (ver loadAllSamples).
    y tem shape (n,) com a saída (gravidade contínua ou classe inteira).
    """
    X = np.array([s.attributesList for s in samples], dtype=float)
    y = np.array([s.output for s in samples], dtype=float)
    return X, y

# K-fold avaliadores — todos retornam (mean_MSE, std_MSE) na escala original

def _classesForStratification(allSamples: list[Sample]) -> np.ndarray:
    """Deriva as classes a partir da gravidade real (via threshold 25/50/75)
    para estratificar os folds, garantindo representação balanceada das 4 classes.
    """
    y_orig = np.array([s.output for s in allSamples], dtype=float)
    return classe_via_regressao(y_orig)

def kFoldDecisionTreeRegressor(allSamples: list[Sample], cv: int = 5) -> tuple[float, float]:
    """K-fold cross-validation do Decision Tree Regressor.

    Returns:
        (mean_MSE, std_MSE) na escala original da gravidade.
    """
    y_cls = _classesForStratification(allSamples)
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)

    fold_mses = []
    for train_idx, test_idx in skf.split(allSamples, y_cls):
        train_samples = [allSamples[i] for i in train_idx]
        test_samples = [allSamples[i] for i in test_idx]

        tree = DecisionTreeRegressor()
        tree.buildTree(train_samples)
        fold_mses.append(tree.testTree(test_samples))

    return float(np.mean(fold_mses)), float(np.std(fold_mses))

def kFoldRandomForestRegressor(allSamples: list[Sample], cv: int = 5) -> tuple[float, float]:
    """K-fold cross-validation do Random Forest Regressor.

    Returns:
        (mean_MSE, std_MSE) na escala original da gravidade.
    """
    y_cls = _classesForStratification(allSamples)
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)

    fold_mses = []
    for train_idx, test_idx in skf.split(allSamples, y_cls):
        train_samples = [allSamples[i] for i in train_idx]
        test_samples = [allSamples[i] for i in test_idx]

        forest = RandomFlorestRegressor()
        forest.buildFlorest(train_samples)
        fold_mses.append(forest.testFlorest(test_samples))

    return float(np.mean(fold_mses)), float(np.std(fold_mses))

def kFoldMLPRegressor(allSamples: list[Sample], cv: int = 5) -> tuple[float, float]:
    """K-fold cross-validation do MLP Regressor.

    Arquitetura [5, 8, 5, 3, 1]:
        - 3 camadas ocultas em funil estreito (8 → 5 → 3 neurônios)
        - 1 saída linear (gravidade contínua)
        - 115 parâmetros — escolhida via ablação de capacidade

    Hiperparâmetros: lr=0.05, momentum=0.9, l2_lambda=0.0

    Returns:
        (mean_MSE, std_MSE) na escala original da gravidade, comparável
        diretamente com o MSE das árvores.
    """
    # Converte samples → arrays (já com FE aplicada em loadAllSamples)
    X, y_orig = samplesToArrays(allSamples)
    y_orig_2d = y_orig.reshape(-1, 1)

    # Normaliza globalmente (pequeno leak aceitável; a classe MLP.cross_validate
    # assume dados já normalizados)
    scaler_X = MinMaxScalerSym().fit(X)
    scaler_y = MinMaxScalerSym().fit(y_orig_2d)
    X_n = scaler_X.transform(X)
    y_n = scaler_y.transform(y_orig_2d)

    mlp = MLP(
        layer_sizes=[5, 8, 5, 3, 1],
        task="regression",
        lr=0.05,
        momentum=0.9,
        seed=42,
        l2_lambda=0.0,                          # L2 não trouxe ganho
    )
    cv_results = mlp.cross_validate(
        X_n, y_n, cv=cv,
        stratify_y=_classesForStratification(allSamples),
        epochs=400, batch_size=32, patience=30,
        verbose=False,
    )
    # Converte MSE da escala normalizada para a original
    #    y_orig - ŷ_orig = (y_norm - ŷ_norm) * range / 2
    #    MSE_orig = MSE_norm * (range / 2)²
    scale = (float(scaler_y.range_[0]) / 2.0) ** 2
    fold_mses_orig = [loss * scale for loss in cv_results['fold_losses']]

    return float(np.mean(fold_mses_orig)), float(np.std(fold_mses_orig))

# Execução principal

if __name__ == "__main__":
    allSamples = loadAllSamples("regressor")

    print(f"K-fold cross-validation (n={len(allSamples)}, 5 folds estratificados)")
    print(f"Features: [qPA, pulso, resp, |pulso-80|, qPA²]")
    print("=" * 60)

    tree_mean, tree_std = kFoldDecisionTreeRegressor(allSamples, cv=5)
    print(f"Decision Tree Regressor   MSE: {tree_mean:.4f} ± {tree_std:.4f}")

    forest_mean, forest_std = kFoldRandomForestRegressor(allSamples, cv=5)
    print(f"Random Forest Regressor   MSE: {forest_mean:.4f} ± {forest_std:.4f}")

    mlp_mean, mlp_std = kFoldMLPRegressor(allSamples, cv=5)
    print(f"MLP Regressor             MSE: {mlp_mean:.4f} ± {mlp_std:.4f}")
