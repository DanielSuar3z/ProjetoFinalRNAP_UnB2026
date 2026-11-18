"""BNN surrogate: ensemble de MLPs con MC-dropout (numpy puro, sin torch).

Ensemble (Lakshminarayanan 2017) + MC-dropout (Gal & Ghahramani 2016).
AMBAS fuentes de varianza son EPISTEMICAS (incertidumbre sobre los pesos):
  - entre miembros del ensemble (bootstrap + inicializacion distinta)
  - MC-dropout dentro de cada miembro (aproximacion variacional bayesiana)
No hay cabeza de varianza aleatoria (ruido heteroscedastico): el simulador es
determinista, asi que la incertidumbre relevante para la adquisicion es la
epistemica. sigma(x) = desviacion de las n_models*mc_samples predicciones.
Devuelve (mu, sigma) en la escala ORIGINAL de y.
Nota: es la version pedagogica/local del modelo. En MC2 puede sustituirse por
el implementado en torch (mismas firmas predict/fit) sin tocar el laço BO.
"""
from __future__ import annotations

import numpy as np


def _relu(x):
    return np.maximum(x, 0.0)


class _MLP:
    """MLP [d -> h1 -> h2 -> 1] con ReLU y dropout en entrenamiento."""

    def __init__(self, d: int, h=(32, 32), seed: int = 0, p_drop: float = 0.05):
        rng = np.random.default_rng(seed)
        sizes = [d, *h, 1]
        self.W = [rng.normal(0, np.sqrt(2.0 / (sizes[i] + sizes[i + 1])),
                             (sizes[i], sizes[i + 1])) for i in range(len(sizes) - 1)]
        self.b = [np.zeros((1, s)) for s in sizes[1:]]
        self.p_drop = p_drop
        self._rng = rng
        # estado Adam
        self.mW = [np.zeros_like(W) for W in self.W]
        self.vW = [np.zeros_like(W) for W in self.W]
        self.mb = [np.zeros_like(b) for b in self.b]
        self.vb = [np.zeros_like(b) for b in self.b]
        self._t = 0

    def _forward(self, X, train: bool):
        """Activaciones por capa. Con dropout, acts[i] ya lleva la mascara
        invertida (a = relu(z) * m / (1-p)), asi que acts[i] > 0 <=> (z>0 y m=1)."""
        acts = [X]
        a = X
        for i, (W, b) in enumerate(zip(self.W, self.b)):
            z = a @ W + b
            a = _relu(z) if i < len(self.W) - 1 else z
            if train and i < len(self.W) - 1 and self.p_drop > 0:
                m = (self._rng.random(a.shape) > self.p_drop)
                a = a * m / (1.0 - self.p_drop)
            acts.append(a)
        return acts

    def predict(self, X, train: bool = False) -> np.ndarray:
        return self._forward(X, train)[-1].ravel()

    def fit(self, X, y, epochs=400, lr=3e-3, batch=64, wd=1e-4, verbose=False):
        n = X.shape[0]
        b1, b2, eps = 0.9, 0.999, 1e-8
        for ep in range(epochs):
            idx = self._rng.permutation(n)
            for s in range(0, n, batch):
                bidx = idx[s:s + batch]
                Xb, yb = X[bidx], y[bidx][:, None]
                acts = self._forward(Xb, train=True)
                # backprop
                gW = [np.zeros_like(W) for W in self.W]
                gb = [np.zeros_like(b) for b in self.b]
                delta = (acts[-1] - yb) / max(n, 1) * 2.0
                for i in range(len(self.W) - 1, -1, -1):
                    a_prev = acts[i]
                    gW[i] = a_prev.T @ delta + wd * self.W[i]
                    gb[i] = delta.mean(axis=0, keepdims=True)
                    if i > 0:
                        # d a / d z = 1[z>0] * m / (1-p): la indicadora sale de
                        # acts[i] > 0 (ya enmascarado) y el factor de escala del
                        # dropout invertido se aplica explicito. (Antes se perdia
                        # el 1/(1-p): la rama de la tupla nunca se ejecutaba.)
                        delta = (delta @ self.W[i].T) * (acts[i] > 0)
                        if self.p_drop > 0:
                            delta = delta / (1.0 - self.p_drop)
                self._t += 1
                for i in range(len(self.W)):
                    for P, mP, vP, g in ((self.W[i], self.mW[i], self.vW[i], gW[i]),
                                         (self.b[i], self.mb[i], self.vb[i], gb[i])):
                        mP[:] = b1 * mP + (1 - b1) * g
                        vP[:] = b2 * vP + (1 - b2) * g ** 2
                        mh = mP / (1 - b1 ** self._t)
                        vh = vP / (1 - b2 ** self._t)
                        P -= lr * mh / (np.sqrt(vh) + eps)
        return self


class BnnEnsemble:
    """n_models MLPs entrenados con bootstrap + MC-dropout en inferencia."""

    def __init__(self, n_models: int = 5, hidden=(32, 32), p_drop=0.05, seed=0):
        self.n_models = n_models
        self.hidden = hidden
        self.p_drop = p_drop
        self.seed = seed
        self.models: list[_MLP] = []
        self.y_mu, self.y_sd = 0.0, 1.0
        self.x_mu, self.x_sd = None, None

    def fit_norm_stats(self, X: np.ndarray):
        self.x_mu = X.mean(axis=0)
        self.x_sd = X.std(axis=0) + 1e-12
        return self

    def _norm(self, X):
        if self.x_mu is None:
            raise RuntimeError("llamar fit_norm_stats(X) antes de fit/predict")
        return (X - self.x_mu) / self.x_sd

    def fit(self, X: np.ndarray, y: np.ndarray, epochs=300, lr=3e-3):
        rng = np.random.default_rng(self.seed)
        self.y_mu, self.y_sd = float(y.mean()), float(y.std() + 1e-12)
        yn = (y - self.y_mu) / self.y_sd
        Xn = self._norm(X)
        self.models = []
        for k in range(self.n_models):
            n = Xn.shape[0]
            idx = rng.integers(0, n, size=n)          # bootstrap
            m = _MLP(Xn.shape[1], self.hidden, seed=self.seed * 100 + k,
                     p_drop=self.p_drop)
            m.fit(Xn[idx], yn[idx], epochs=epochs, lr=lr)
            self.models.append(m)
        return self

    def predict(self, X: np.ndarray, mc_samples: int = 16):
        """(mu, sigma) en la escala original de y. Epistemica + MC-dropout."""
        Xn = self._norm(X)
        preds = []
        for m in self.models:
            for _ in range(mc_samples):               # MC dropout
                preds.append(m.predict(Xn, train=True))
        P = np.vstack(preds)                          # (n_models*mc, n_pts)
        mu = P.mean(axis=0) * self.y_sd + self.y_mu
        sd = P.std(axis=0) * self.y_sd
        return mu, sd

    # ---- persistencia (contrato bnn_v1.npz: lo consume resbo_wellcontrol/T-R5) --
    def save_npz(self, path: str) -> str:
        """Pesos + estadisticos de normalizacion en un npz sin pickle.
        Claves: x_mu/x_sd/y_mu/y_sd/hidden/p_drop/seed/n_models y m{k}_W{i}/m{k}_b{i}."""
        if not self.models:
            raise RuntimeError("nada que guardar: entrenar el ensemble primero")
        d = {"x_mu": self.x_mu, "x_sd": self.x_sd,
             "y_mu": np.float64(self.y_mu), "y_sd": np.float64(self.y_sd),
             "hidden": np.asarray(self.hidden, dtype="int64"),
             "p_drop": np.float64(self.p_drop), "seed": np.int64(self.seed),
             "n_models": np.int64(self.n_models)}
        for k, m in enumerate(self.models):
            for i, (W, b) in enumerate(zip(m.W, m.b)):
                d[f"m{k}_W{i}"] = W
                d[f"m{k}_b{i}"] = b
        np.savez_compressed(path, **d)
        return path

    @classmethod
    def load_npz(cls, path: str) -> "BnnEnsemble":
        """Reconstruye el ensemble de save_npz (allow_pickle=False)."""
        z = np.load(path, allow_pickle=False)
        hidden = tuple(int(v) for v in z["hidden"])
        obj = cls(n_models=int(z["n_models"]), hidden=hidden,
                  p_drop=float(z["p_drop"]), seed=int(z["seed"]))
        obj.x_mu, obj.x_sd = z["x_mu"], z["x_sd"]
        obj.y_mu, obj.y_sd = float(z["y_mu"]), float(z["y_sd"])
        d = len(z["x_mu"])
        obj.models = []
        for k in range(obj.n_models):
            m = _MLP(d, hidden, seed=0, p_drop=obj.p_drop)
            m.W = [np.asarray(z[f"m{k}_W{i}"]) for i in range(len(m.W))]
            m.b = [np.asarray(z[f"m{k}_b{i}"]) for i in range(len(m.b))]
            obj.models.append(m)
        return obj
