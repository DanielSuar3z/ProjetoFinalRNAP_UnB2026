# Guía: `src/resbo/config.py` — el loader del protocolo

**Rol de una línea:** lee `configs/protocol.yaml` y lo expone como un objeto
`Protocol` con accesos con nombre (`proto.n_inj`, `proto.bounds`), para que
ningún otro módulo vuelva a escribir la ruta del YAML ni los literales 8/4/130.

**Etiqueta de dominio:** "protocolo" = el conjunto de reglas idénticas para
todos los brazos (caja de controles, economía, presupuestos). Se llama así
porque es literalmente el protocolo experimental del proyecto; cambiarlo es
cambiar el experimento, no el código.

---

## `class Protocol`

Envoltorio ("wrapper") fino sobre el dict que produce `yaml.safe_load`. No
valida ni copia: es azúcar de acceso. Se llama `Protocol` (no `Config`) para
reforzar que **es el protocolo experimental del artículo**, no una config
cualquiera de software.

### `__init__(self, d: dict)`

- `d` — el dict plano ya parseado del YAML. Se guarda tal cual en `self.d`.
  Se mantiene público (sin guion bajo) porque los scripts lo tocan
  deliberadamente: `toy_experiment.py --quick` reesponde presupuestos haciendo
  `proto.d["budget"]["bo"] = {...}`. Un acceso mutante por diseño, no un descuido.

### `@classmethod load(cls, path) -> Protocol`

- **Qué hace:** abre el archivo y hace `Protocol(yaml.safe_load(f))`.
- `path` — ruta al YAML. `yaml.safe_load` (en vez de `yaml.load`) porque el
  parser "safe" nunca ejecuta etiquetas de Python arbitrarias del archivo:
  seguridad básica aunque el YAML sea nuestro.
- `@classmethod` (no función suelta) porque es el constructor alternativo
  idiomático de Python: `Protocol.load(...)` se lee como "construir desde disco".

### `__getitem__(self, k)`

- Define el operador `proto["budget"]` → delega en `self.d[k]`.
- Por qué existe: permite la doble sintaxis — accesos frecuentes con
  atributo (`proto.n_inj`) y accesos estructurales crudos (`proto["prior"]["w0"]`)
  sin escribir `proto.d` por todos lados.

### `@property n_inj` / `n_prod` / `horizon`

| Propiedad | Clave YAML | Qué es | Por qué el nombre |
|---|---|---|---|
| `n_inj` | `n_injectors` | nº de inyectores (Egg: 8) | `n_` = count, convención científica; `inj` = injector |
| `n_prod` | `n_producers` | nº de productores (Egg: 4) | ídem, `prod` = producer |
| `horizon` | `horizon_months` | vida útil simulada en meses (130) | "horizon" = horizonte de evaluación; cada simulación cubre estos meses |

`int(...)` convierte por si el YAML se leyó como string o float. Se usan como
`@property` porque son derivaciones constantes del dict: acceso de atributo es
más legible que método.

### `@property bounds -> (lb, ub)`

El método más importante del archivo: **convierte la descripción escalar de la
caja en vectores de 12 dimensiones**.

```python
r = self.d["bounds"]["rate"]   # [500.0, 3000.0]  (sm3/d)
b = self.d["bounds"]["bhp"]    # [150.0, 350.0]   (bar)
lb = np.array([r[0]] * n_inj + [b[0]] * n_prod)   # 8×500, 4×150 → (12,)
ub = np.array([r[1]] * n_inj + [b[1]] * n_prod)   # 8×3000, 4×350 → (12,)
```

- `lb` / `ub` — **l**ower/**u**pper **b**ound. Son la firma de la caja: todo
  algoritmo hace `np.clip(X, lb, ub)` con ellos.
- El orden del vector (rates primero, BHP después) **define la convención del
  proyecto**: los algoritmos parten poblaciones con `lb + rand*(ub-lb)` y el
  simulador corta `controls[:n_inj]` / `controls[n_inj:]`. Si algún día se
  invierte el orden, se rompe todo y en silencio — por eso está en un único sitio.
- Nota: `import numpy` está dentro del método para que el módulo se importe
  rápido aunque solo se use `Protocol.load`; es optimización cosmética.

### `@property n_seeds`

- Nº de repeticiones (semillas) por brazo. Demo toy: 4; protocolo del
  artículo: 20 (necesario para el Mann-Whitney con potencia razonable).

---

## `repo_root() -> str`

- Devuelve la carpeta raíz `resbo/` calculándola **desde la ubicación del
  propio archivo** (`__file__` → subir dos niveles: `src/resbo/` → `resbo/`),
  no desde el directorio de trabajo actual.
- Por qué: permite lanzar scripts desde cualquier carpeta sin romper rutas.
- Actualmente `toy_experiment.py` calcula su propia raíz (hace lo mismo desde
  `scripts/`); esta función queda para los scripts futuros de MC2.

---

## Cómo se conecta con el resto

`Protocol` viaja por parámetro a **todos** los objetos del proyecto (MFO, GWO,
PSO, FOA, BayesOptimizer, MockSimulator, CRM…). Ningún módulo lee el YAML por
su cuenta: si un día el protocolo cambia de formato, solo se toca este archivo.
