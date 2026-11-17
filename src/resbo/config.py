"""Config loader: lee configs/protocol.yaml (dict plano con anidamiento simple)."""
from __future__ import annotations
import os

import yaml


class Protocol:
    """Envoltorio de acceso comodo sobre el YAML del protocolo."""

    def __init__(self, d: dict):
        self.d = d

    @classmethod
    def load(cls, path: str) -> "Protocol":
        with open(path) as f:
            return cls(yaml.safe_load(f))

    def __getitem__(self, k):
        return self.d[k]

    @property
    def n_inj(self) -> int:
        return int(self.d["n_injectors"])

    @property
    def n_prod(self) -> int:
        return int(self.d["n_producers"])

    @property
    def horizon(self) -> int:
        return int(self.d["horizon_months"])

    @property
    def bounds(self):
        """(lb, ub) del vector de control [rate x n_inj, bhp x n_prod]."""
        r = self.d["bounds"]["rate"]
        b = self.d["bounds"]["bhp"]
        import numpy as np
        lb = np.array([r[0]] * self.n_inj + [b[0]] * self.n_prod)
        ub = np.array([r[1]] * self.n_inj + [b[1]] * self.n_prod)
        return lb, ub

    @property
    def n_seeds(self) -> int:
        return int(self.d["n_seeds"])


    @property
    def base_controls(self):
        """Control base del deck [rate x n_inj, bhp x n_prod] (injectorValues/
        producerValues de BIOPSO). Sin la seccion `base_controls`, centro de caja."""
        import numpy as np
        bc = self.d.get("base_controls")
        if not bc:
            lb, ub = self.bounds
            return 0.5 * (lb + ub)
        r, b = float(bc["rate"]), float(bc["bhp"])
        return np.array([r] * self.n_inj + [b] * self.n_prod)


def repo_root() -> str:
    """Directorio raiz del proyecto resbo/ (contiene configs/ y experiments/)."""
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
