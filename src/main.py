from constants import *
from sample import Sample
from decisionTreeRegressor import DecisionTreeRegressor
from randomFlorestRegressor import RandomFlorestRegressor
from mlp import MLP, MinMaxScalerSym, classe_via_regressao

from typing import Literal
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
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


# K-fold avaliadores — todos usam StratifiedKFold(random_state=42) garantindo
# que o mesmo Sample caia no MESMO fold em todas as funções (regressor e
# classificador). Isso permite combinar gravidade e classe na mesma linha do CSV.

def _classesForStratification(allSamples_reg: list[Sample]) -> np.ndarray:
    """Deriva as classes a partir da gravidade real
    para estratificar os folds, garantindo representação balanceada das 4 classes.

    Recebe sempre samples do regressor (output=gravidade contínua) para que
    a estratificação seja CONSISTENTE entre todas as funções k-fold.
    """
    y_orig = np.array([s.output for s in allSamples_reg], dtype=float)
    return classe_via_regressao(y_orig)


def kFoldDecisionTreeRegressor(allSamples_reg: list[Sample], cv: int = 5) -> tuple[float, float, np.ndarray]:
    """K-fold do Decision Tree Regressor.

    Returns:
        (mean_MSE, std_MSE, gravidade_oof) — gravidade_oof tem 1 predição
        por amostra (out-of-fold), na escala original.
    """
    y_cls = _classesForStratification(allSamples_reg)
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)

    fold_mses = []
    gravidade_oof = np.zeros(len(allSamples_reg))

    for train_idx, test_idx in skf.split(allSamples_reg, y_cls):
        train_samples = [allSamples_reg[i] for i in train_idx]
        test_samples = [allSamples_reg[i] for i in test_idx]

        tree = DecisionTreeRegressor()
        tree.buildTree(train_samples)
        fold_mses.append(tree.testTree(test_samples))

        for i, s in zip(test_idx, test_samples):
            gravidade_oof[i] = tree.predict(s)

    return float(np.mean(fold_mses)), float(np.std(fold_mses)), gravidade_oof


def kFoldRandomForestRegressor(allSamples_reg: list[Sample], cv: int = 5) -> tuple[float, float, np.ndarray]:
    """K-fold do Random Forest Regressor.

    Returns:
        (mean_MSE, std_MSE, gravidade_oof).
    """
    y_cls = _classesForStratification(allSamples_reg)
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)

    fold_mses = []
    gravidade_oof = np.zeros(len(allSamples_reg))

    for train_idx, test_idx in skf.split(allSamples_reg, y_cls):
        train_samples = [allSamples_reg[i] for i in train_idx]
        test_samples = [allSamples_reg[i] for i in test_idx]

        forest = RandomFlorestRegressor()
        forest.buildFlorest(train_samples)
        fold_mses.append(forest.testFlorest(test_samples))

        for i, s in zip(test_idx, test_samples):
            gravidade_oof[i] = forest.predict(s)

    return float(np.mean(fold_mses)), float(np.std(fold_mses)), gravidade_oof


def kFoldMLPRegressor(allSamples_reg: list[Sample], cv: int = 5) -> tuple[float, float, np.ndarray]:
    """K-fold do MLP Regressor

    Arquitetura [5, 8, 5, 3, 1]:
        - 3 camadas ocultas em funil estreito (8 → 5 → 3 neurônios)
        - 1 saída linear (gravidade contínua)
        - 115 parâmetros — escolhida via ablação de capacidade

    Hiperparâmetros: lr=0.05, momentum=0.9, l2_lambda=0.0

    A classe é derivada por threshold no main (classe_via_regressao) — foi a
    melhor estratégia encontrada nas iterações: regressão + threshold > softmax direto.

    Returns:
        (mean_MSE, std_MSE, gravidade_oof) na escala original da gravidade.
    """
    X, y_orig = samplesToArrays(allSamples_reg)
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
        stratify_y=_classesForStratification(allSamples_reg),
        epochs=400, batch_size=32, patience=30,
        verbose=False,
    )

    # Reconstrói predições OOF em escala original
    gravidade_oof = np.zeros(len(allSamples_reg))
    for fold_preds_n, fold_idx in zip(cv_results['fold_predictions'], cv_results['fold_indices']):
        fold_preds = scaler_y.inverse_transform(fold_preds_n).flatten()
        gravidade_oof[fold_idx] = fold_preds

    # Converte MSE da escala normalizada para a original
    scale = (float(scaler_y.range_[0]) / 2.0) ** 2
    fold_mses_orig = [loss * scale for loss in cv_results['fold_losses']]

    return float(np.mean(fold_mses_orig)), float(np.std(fold_mses_orig)), gravidade_oof


# Gerador de arquivo de saída no formato exigido pelo enunciado

def saveOutputCSV(filename: str, allSamples: list[Sample],
                  gravidade: np.ndarray, classe: np.ndarray) -> None:
    """Salva o arquivo de predições no formato do enunciado: i, gravidade, classe.

    Cada linha tem 3 colunas separadas por vírgulas:
        - i        : identificador da amostra (int)
        - gravidade: predição contínua da gravidade (float, 4 casas decimais)
        - classe   : predição da classe ∈ {1, 2, 3, 4} (int)
    """
    with open(filename, "w") as f:
        for s, g, c in zip(allSamples, gravidade, classe):
            f.write(f"{s.id}, {g:.4f}, {int(c)}\n")


# Execução principal

if __name__ == "__main__":
    allSamples_reg = loadAllSamples("regressor")
    allSamples_cls = loadAllSamples("classifier")
    y_true_cls = np.array([s.output for s in allSamples_cls], dtype=int)

    print(f"K-fold cross-validation (n={len(allSamples_reg)}, 5 folds estratificados)")
    print(f"Features: [qPA, pulso, resp, |pulso-80|, qPA²]")
    print("=" * 80)

    # Decision Tree — regressor; classe derivada via classe_via_regressao
    tree_mse, tree_mse_std, tree_grav = kFoldDecisionTreeRegressor(allSamples_reg, cv=5)
    tree_cls = classe_via_regressao(tree_grav)
    tree_acc = float((tree_cls == y_true_cls).mean())
    tree_f1m = float(f1_score(y_true_cls, tree_cls, average="macro"))
    print(f"Decision Tree   MSE: {tree_mse:6.4f} ± {tree_mse_std:.4f}  |  Acc: {tree_acc:.4f}  |  F1 macro: {tree_f1m:.4f}")
    saveOutputCSV("predicoes_tree.csv", allSamples_reg, tree_grav, tree_cls)

    # Random Forest — regressor; classe derivada via classe_via_regressao
    forest_mse, forest_mse_std, forest_grav = kFoldRandomForestRegressor(allSamples_reg, cv=5)
    forest_cls = classe_via_regressao(forest_grav)
    forest_acc = float((forest_cls == y_true_cls).mean())
    forest_f1m = float(f1_score(y_true_cls, forest_cls, average="macro"))
    print(f"Random Forest   MSE: {forest_mse:6.4f} ± {forest_mse_std:.4f}  |  Acc: {forest_acc:.4f}  |  F1 macro: {forest_f1m:.4f}")
    saveOutputCSV("predicoes_forest.csv", allSamples_reg, forest_grav, forest_cls)

    # MLP — regressor; classe derivada via classe_via_regressao
    mlp_mse, mlp_mse_std, mlp_grav = kFoldMLPRegressor(allSamples_reg, cv=5)
    mlp_cls = classe_via_regressao(mlp_grav)
    mlp_acc = float((mlp_cls == y_true_cls).mean())
    mlp_f1m = float(f1_score(y_true_cls, mlp_cls, average="macro"))
    print(f"MLP             MSE: {mlp_mse:6.4f} ± {mlp_mse_std:.4f}  |  Acc: {mlp_acc:.4f}  |  F1 macro: {mlp_f1m:.4f}")
    saveOutputCSV("predicoes_mlp.csv", allSamples_reg, mlp_grav, mlp_cls)

    print()
    print("Arquivos gerados:")
    print("  predicoes_tree.csv")
    print("  predicoes_forest.csv")
    print("  predicoes_mlp.csv")
