"""

Implementação de um MLP para a disciplina de Sistemas Inteligentes
Construída para o problema de classificação/regressão de vítimas de catástrofes

Componentes:
    - MLP                   Classe principal do MLP
    - MinMaxScalerSym       Normalizador min-max para o intervalo [-1, +1]
    - classe_via_regressao  Conversão de saída de regressão para classe via threshold

Capacidades implementadas:
    - Backpropagation com gradiente da regra da cadeia
    - Otimização SGD mini-batch + momentum
    - Suporte a regressão e classificação
    - Pesos por classe na cross-entropy
    - Regularização L2
    - Early stopping pelo erro de validação
    - K-fold cross-validation como método da classe

Uso típico:

    from mlp import MLP, MinMaxScalerSym, classe_via_regressao

    # Normalizar features
    scaler = MinMaxScalerSym().fit(X_train)
    X_train_n = scaler.transform(X_train)

    # Treinar regressor
    mlp = MLP(layer_sizes=[5, 8, 5, 3, 1], task='regression',
              lr=0.05, momentum=0.9, l2_lambda=0.0, seed=42)
    mlp.fit(X_train_n, y_train_n, X_val=X_val_n, y_val=y_val_n,
            epochs=400, batch_size=32, patience=30)

    # K-fold cross-validation
    scores = mlp.cross_validate(X_train_n, y_train_n, cv=5)
    print(f"MSE médio: {scores['mean_loss']:.4f} ± {scores['std_loss']:.4f}")
"""

import numpy as np


class MinMaxScalerSym:
    """Min-max scaler para o intervalo simétrico [-1, +1].

    Ajusta min e max por coluna do treino e usa essas estatísticas para
    transformar quaisquer dados subsequentes (val/teste). O `fit` é feito
    apenas no treino para evitar data leakage

    Fórmula: x_norm = 2 * (x - min) / (max - min) - 1
    """

    def fit(self, X):
        X = np.asarray(X)
        self.min_ = X.min(axis=0)
        self.max_ = X.max(axis=0)
        self.range_ = self.max_ - self.min_
        # Evita divisão por zero em features constantes
        self.range_ = np.where(self.range_ == 0, 1.0, self.range_)
        return self

    def transform(self, X):
        return 2.0 * (np.asarray(X) - self.min_) / self.range_ - 1.0

    def fit_transform(self, X):
        return self.fit(X).transform(X)

    def inverse_transform(self, Xn):
        return (np.asarray(Xn) + 1.0) * self.range_ / 2.0 + self.min_


def classe_via_regressao(gravidade_pred, thresholds=(25, 50, 75)):
    """Mapeia uma gravidade prevista para classe {1, 2, 3, 4}
    usando os thresholds avaliados na exploração do dataset

    np.digitize retorna 0/1/2/3 conforme o valor está em (-inf, 25), [25, 50),
    [50, 75) ou [75, +inf). Somamos 1 para obter as classes 1/2/3/4

    Args:
        gravidade_pred: array de gravidades preditas
        thresholds: tupla com os limiares (default: 25, 50, 75) Explorados
                    no dataset de vítimas

    Returns:
        Array 1-D com classes em {1, 2, 3, 4}.
    """
    g = np.asarray(gravidade_pred).flatten()
    return np.digitize(g, bins=list(thresholds)) + 1


class MLP:
    """
    Arquitetura: camadas densas com ativação `tanh` nas ocultas e
    `linear` ou `softmax` na saída

    SGD mini-batch com momentum (β=0 desativa momentum)
    Suporta L2, early stopping, pesos por classe e
    k-fold cross-validation como método

    Args:
        layer_sizes (list[int]): dimensões `[in, h1, h2, ..., out]`.
                                  Para regressão use out=1, classificação use out=n_classes.
        task (str): 'regression' (MSE + linear) ou 'classification' (CE + softmax).
        lr (float): taxa de aprendizado.
        momentum (float): coeficiente β do momentum (0 = sem momentum).
        seed (int): seed para inicialização determinística dos pesos.
        l2_lambda (float): coeficiente de regularização L2 (0 = sem L2).
    """

    def __init__(
        self,
        layer_sizes,
        task='regression',
        lr=0.01,
        momentum=0.9,
        seed=42,
        l2_lambda=0.0,
    ):
        assert task in ('regression', 'classification'), \
            "task deve ser 'regression' ou 'classification'"
        assert len(layer_sizes) >= 2, "É preciso pelo menos entrada e saída"

        self.layer_sizes = list(layer_sizes)
        self.task = task
        self.lr = lr
        self.momentum = momentum
        self.seed = seed
        self.l2_lambda = l2_lambda

        # Inicialização aleatória uniforme U(-0.5, +0.5) faixa mais conservadora adotada, em comparação a -1,+1
        rng = np.random.RandomState(seed)
        self.weights, self.biases = [], []
        self.v_w, self.v_b = [], []
        for i in range(len(layer_sizes) - 1):
            w = rng.uniform(-0.5, 0.5, size=(layer_sizes[i], layer_sizes[i + 1]))
            b = np.zeros(layer_sizes[i + 1])
            self.weights.append(w)
            self.biases.append(b)
            self.v_w.append(np.zeros_like(w))
            self.v_b.append(np.zeros_like(b))

    # Funções de ativação
    @staticmethod
    def _tanh(z):
        return np.tanh(z)

    @staticmethod
    def _tanh_deriv_from_act(a):
        # Dada a ativação a = tanh(z), a derivada é 1 - a²
        return 1.0 - a * a

    @staticmethod
    def _softmax(z):
        # Softmax numericamente estável, (subtrai o máximo por linha)
        z_shift = z - z.max(axis=1, keepdims=True)
        e = np.exp(z_shift)
        return e / e.sum(axis=1, keepdims=True)

    # Forward pass
    def _forward(self, X):
        """Propaga entrada X pelas camadas e retorna a saída
        Armazena ativações intermediárias em self.cache_a para uso no backward.
        """
        self.cache_a = [X]
        a = X
        L = len(self.weights)
        for i in range(L):
            z = a @ self.weights[i] + self.biases[i]
            if i < L - 1:
                # Camadas ocultas usam tanh
                a = self._tanh(z)
            else:
                # Camada de saída: linear (regressão) ou softmax (classificação)
                a = z if self.task == 'regression' else self._softmax(z)
            self.cache_a.append(a)
        return a

    # backpropagation
    def _backward(self, y_true, sample_weights=None):
        """Calcula os gradientes dos pesos e biases
        Args:
            y_true: target (one-hot para classificação, valor real para regressão).
            sample_weights: pesos opcionais por amostra (shape (m,)).
                           Quando passado, multiplica `dz` da saída linha a linha.
        """
        m = y_true.shape[0]
        L = len(self.weights)
        grads_w = [None] * L
        grads_b = [None] * L

        # Gradiente na saída:
        # - Regressão (MSE + linear): dL/dz = (ŷ - y)
        # - Classificação (CE + softmax): dL/dz = (softmax - y_onehot)
        dz = self.cache_a[-1] - y_true

        # Aplicar pesos por amostra (para cost-sensitive learning)
        if sample_weights is not None:
            dz = dz * sample_weights[:, None]

        # Retropropagação camada a camada
        for i in reversed(range(L)):
            grads_w[i] = self.cache_a[i].T @ dz / m
            # L2 regularization: penaliza apenas pesos, não biases
            if self.l2_lambda > 0:
                grads_w[i] = grads_w[i] + 2.0 * self.l2_lambda * self.weights[i]
            grads_b[i] = dz.mean(axis=0)
            if i > 0:
                # Propaga gradiente para camada anterior usando regra da cadeia
                da = dz @ self.weights[i].T
                dz = da * self._tanh_deriv_from_act(self.cache_a[i])

        return grads_w, grads_b

    # Atualização de pesos com momentum
    def _step(self, grads_w, grads_b):
        """Atualiza pesos e biases usando o gradiente e o momentum acumulado

        Equação: v := β·v - lr·gradiente ; w := w + v
        """
        for i in range(len(self.weights)):
            self.v_w[i] = self.momentum * self.v_w[i] - self.lr * grads_w[i]
            self.v_b[i] = self.momentum * self.v_b[i] - self.lr * grads_b[i]
            self.weights[i] += self.v_w[i]
            self.biases[i] += self.v_b[i]

    # Função de perda
    def _loss(self, y_pred, y_true, sample_weights=None):
        """Calcula a perda média (MSE para regressão, CE para classificação)."""
        if self.task == 'regression':
            err = (y_pred - y_true) ** 2
            if sample_weights is not None:
                err = err * sample_weights[:, None]
            return float(err.mean())
        # Cross-entropy categórica (com epsilon para estabilidade numérica)
        eps = 1e-12
        per_sample = -np.sum(y_true * np.log(y_pred + eps), axis=1)
        if sample_weights is not None:
            per_sample = per_sample * sample_weights
        return float(per_sample.mean())

    def _sample_weights_from_class(self, y_oh, class_weights):
        """Converte vetor de pesos por classe (K,) em pesos por amostra (m,)
        usando o rótulo verdadeiro one-hot. Retorna None se class_weights for None."""
        if class_weights is None:
            return None
        y_idx = y_oh.argmax(axis=1)
        return class_weights[y_idx]

    # Treinamento
    def fit(
        self,
        X_tr, y_tr,
        X_val=None, y_val=None,
        epochs=300, batch_size=32, patience=25,
        verbose=False,
        shuffle_seed=0,
        class_weights=None,
    ):
        """Treina o MLP usando SGD mini-batch + momentum.

        Args:
            X_tr, y_tr: features e targets de treino.
            X_val, y_val: opcional — usados para early stopping (paciência configurável).
            epochs: máximo de épocas.
            batch_size: tamanho do mini-batch.
            patience: épocas de paciência para early stopping.
            verbose: imprime progresso a cada 25 épocas.
            shuffle_seed: seed para shuffle dos batches (reprodutibilidade).
            class_weights: opcional (shape (K,)) — para classificação ponderada.

        Returns:
            dict com chaves 'train_loss', 'val_loss', 'grad_norm'.
        """
        history = {'train_loss': [], 'val_loss': [], 'grad_norm': []}
        rng = np.random.RandomState(shuffle_seed)
        n = X_tr.shape[0]
        best_val = float('inf')
        best_state = None
        bad = 0

        # Pre-calcula sample_weights se for classificação ponderada
        sw_tr = (
            self._sample_weights_from_class(y_tr, class_weights)
            if self.task == 'classification' else None
        )
        sw_val = (
            self._sample_weights_from_class(y_val, class_weights)
            if (self.task == 'classification' and y_val is not None) else None
        )

        for epoch in range(epochs):
            idx = rng.permutation(n)
            Xs, ys = X_tr[idx], y_tr[idx]
            sws = sw_tr[idx] if sw_tr is not None else None
            gnorms = []

            for start in range(0, n, batch_size):
                Xb = Xs[start:start + batch_size]
                yb = ys[start:start + batch_size]
                swb = sws[start:start + batch_size] if sws is not None else None
                self._forward(Xb)
                gw, gb = self._backward(yb, sample_weights=swb)
                gn = np.sqrt(
                    sum((g * g).sum() for g in gw) + sum((g * g).sum() for g in gb)
                )
                gnorms.append(gn)
                self._step(gw, gb)

            train_loss = self._loss(self._forward(X_tr), y_tr, sample_weights=sw_tr)
            history['train_loss'].append(train_loss)
            history['grad_norm'].append(float(np.mean(gnorms)))

            if X_val is not None:
                val_loss = self._loss(self._forward(X_val), y_val, sample_weights=sw_val)
                history['val_loss'].append(val_loss)
                if val_loss < best_val - 1e-6:
                    best_val = val_loss
                    best_state = (
                        [w.copy() for w in self.weights],
                        [b.copy() for b in self.biases],
                    )
                    bad = 0
                else:
                    bad += 1
                    if bad >= patience:
                        if verbose:
                            print(f'  early stopping na época {epoch + 1}')
                        break

            if verbose and (epoch + 1) % 25 == 0:
                msg = f'  ep {epoch + 1:3d}  train={train_loss:.4f}'
                if X_val is not None:
                    msg += f'  val={val_loss:.4f}'
                print(msg)

        # Restaura os melhores pesos vistos durante o treino
        if best_state is not None:
            self.weights, self.biases = best_state
        return history

    # Inferência
    def predict(self, X):
        """Retorna a saída do forward pass (gravidade para regressão, probs para classificação)."""
        return self._forward(X)

    def predict_classes(self, X):
        """Para classificação: retorna a classe via argmax + 1 (classes 1..K)."""
        assert self.task == 'classification', "Use predict() para regressão"
        probs = self._forward(X)
        return probs.argmax(axis=1) + 1

    # K-fold cross-validation
    def cross_validate(
        self,
        X, y,
        cv=5,
        stratify_y=None,
        val_frac=0.125,
        epochs=400,
        batch_size=32,
        patience=30,
        verbose=False,
    ):
        """K-fold cross-validation usando os hiperparâmetros desta instância.

        Cria `cv` modelos novos (com a mesma arquitetura, lr, momentum, l2_lambda),
        treina cada um em k-1 folds e avalia no fold restante.

        Cada fold usa uma seed independente derivada do `self.seed` da instância.
        a chamada continua reproduzivel, mesma seed gera mesmas
        métricas, mas cada fold tem inicialização própria
        o `std_loss` retornado captura variância dos dados + otimização, e não
        apenas dos dados

        Args:
            X, y: features e targets (já normalizados/encodados conforme `task`).
            cv: número de folds (default 5).
            stratify_y: array opcional de rótulos para estratificação. Se None,
                       usa KFold normal; se fornecido, usa StratifiedKFold com
                       essa estratificação.
            val_frac: fração do treino-do-fold usada para validação (early stopping).
            epochs, batch_size, patience: parâmetros do fit em cada fold.
            verbose: se True, imprime progresso fold a fold.

        Returns:
            dict com:
                - 'mean_loss': float, média das perdas no fold de teste
                - 'std_loss': float, desvio padrão (dados + otimização)
                - 'fold_losses': list[float], perda por fold
                - 'fold_predictions': list[np.ndarray], predições out-of-fold
                - 'fold_indices': list[np.ndarray], índices de teste por fold
                - 'fold_seeds': list[int], seed usada em cada fold (debug)
        """
        # Import local para não exigir sklearn no nível do módulo
        try:
            from sklearn.model_selection import KFold, StratifiedKFold
        except ImportError as exc:
            raise ImportError(
                "cross_validate requer scikit-learn. Instale com `pip install scikit-learn`."
            ) from exc

        if stratify_y is not None:
            splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=self.seed)
            splits = list(splitter.split(X, stratify_y))
        else:
            splitter = KFold(n_splits=cv, shuffle=True, random_state=self.seed)
            splits = list(splitter.split(X))

        # Deriva uma seed independente por fold a partir do master seed
        # Cada fold terá inicialização própria, mas o resultado global continua
        # reproduzível
        master_rng = np.random.RandomState(self.seed)
        fold_seeds = master_rng.randint(0, 100_000, size=cv).tolist()

        fold_losses = []
        fold_predictions = []
        fold_indices = []

        for fold_idx, (train_idx, test_idx) in enumerate(splits):
            fold_seed = int(fold_seeds[fold_idx])

            # Subdivide treino do fold em treino + validação
            # Usa fold_seed para que o split val/train também varie por fold.
            rng = np.random.RandomState(fold_seed)
            perm = rng.permutation(train_idx)
            n_val = int(val_frac * len(perm))
            val_idx = perm[:n_val]
            tr_idx = perm[n_val:]

            # Cria novo modelo com os mesmos hiperparâmetros desta instancia,
            # mas com seed específica do fold
            m = MLP(
                layer_sizes=self.layer_sizes,
                task=self.task,
                lr=self.lr,
                momentum=self.momentum,
                seed=fold_seed,
                l2_lambda=self.l2_lambda,
            )
            m.fit(
                X[tr_idx], y[tr_idx],
                X_val=X[val_idx], y_val=y[val_idx],
                epochs=epochs, batch_size=batch_size, patience=patience,
                shuffle_seed=fold_seed, verbose=False,
            )

            # Avaliar no fold de teste
            y_pred = m.predict(X[test_idx])
            loss = m._loss(y_pred, y[test_idx])

            fold_losses.append(loss)
            fold_predictions.append(y_pred)
            fold_indices.append(test_idx)

            if verbose:
                print(f'  Fold {fold_idx + 1}/{cv}: loss = {loss:.4f}')

        return {
            'mean_loss': float(np.mean(fold_losses)),
            'std_loss': float(np.std(fold_losses)),
            'fold_losses': fold_losses,
            'fold_predictions': fold_predictions,
            'fold_indices': fold_indices,
            'fold_seeds': fold_seeds,
        }
