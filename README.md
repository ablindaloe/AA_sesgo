# Análisis de equidad y transparencia en la predicción de la reincidencia

Proyecto de la asignatura **Aprendizaje Avanzado** (3.º curso, Ingeniería en IA) sobre el sesgo intrínseco del dataset judicial **COMPAS**.

**Autores:** Stefania Bianchi · Alex Garre · Carla Milán · Alejandro Conesa

---

## Contenido del repositorio

```
repo/
├── proyecto_aa_sesgo.ipynb   # Notebook principal con todo el análisis
├── requirements.txt          # Dependencias
├── images/                   # Figuras generadas
└── README.md
```

El notebook se divide en cinco partes que se ejecutan en orden:

1. **EDA** — Análisis exploratorio del dataset COMPAS (7214 instancias).
2. **Clasificadores estándar** — 5 modelos (LR, SVM-RBF, Random Forest, XGBoost, MLP) con `GridSearchCV` + `StratifiedKFold` 5-fold.
3. **Mitigación de sesgo con Fairlearn** — `CorrelationRemover` (preprocesamiento) y `ThresholdOptimizer` (postprocesamiento).
4. **Auditoría con SHAP** — Interpretabilidad de LR y XGBoost.
5. **Evaluación multidimensional** — Rendimiento, equidad por raza, equidad por sexo y robustez ante ruido.

---

## Datos

El dataset se descarga automáticamente al ejecutar la primera celda desde:

```
https://raw.githubusercontent.com/propublica/compas-analysis/master/compas-scores-two-years.csv
```

Se necesita **conexión a internet** en la primera ejecución.

---

## Reproducibilidad

Para garantizar resultados idénticos a los reportados en la memoria y la presentación, el notebook fija una semilla global y usa validación cruzada estratificada con el mismo `random_state`:

- `RANDOM_STATE = 42` en todas las operaciones estocásticas (`train_test_split`, `StratifiedKFold`, modelos).
- `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)` compartido.
- Split de test: 80/20 estratificado sobre `two_year_recid`.
- Versiones de librerías fijadas en `requirements.txt`.


---

## Ejecución en local

### Opción A — `venv` + `pip` (recomendada)

```bash
# 1. Clonar el repositorio
git clone <url-del-repo>
cd repo

# 2. Crear y activar entorno virtual
python3 -m venv .venv
source .venv/bin/activate          # Linux / macOS
# .venv\Scripts\activate           # Windows (PowerShell)

# 3. Instalar dependencias
pip install --upgrade pip
pip install -r requirements.txt

# 4. Lanzar Jupyter
jupyter notebook proyecto_aa_sesgo.ipynb
```

Dentro de la interfaz de Jupyter: **Kernel → Restart & Run All** ejecuta el notebook completo de principio a fin.

### Opción B — `conda`

```bash
conda create -n aa python=3.12 -y
conda activate aa
pip install -r requirements.txt
jupyter notebook proyecto_aa_sesgo.ipynb
```


---

## Ejecución en Google Colab

1. Abre [colab.research.google.com](https://colab.research.google.com/).
2. **File → Upload notebook** y sube `proyecto_aa_sesgo.ipynb` (o usa **File → Open notebook → GitHub** y pega la URL del repositorio).
3. Añade una celda **al inicio** del notebook con:

   ```python
   !pip install -q -r https://raw.githubusercontent.com/<usuario>/<repo>/main/requirements.txt
   ```

   Si no quieres usar el `requirements.txt` remoto, equivalentemente:

   ```python
   !pip install -q fairlearn==0.12.0 shap==0.46.0 xgboost==3.0.0
   ```

   (Colab ya trae `pandas`, `numpy`, `scikit-learn`, `matplotlib` y `seaborn` con versiones compatibles.)

4. **Runtime → Run all** (`Ctrl + F9`).

> **Nota Colab:** no se necesita GPU; el runtime **CPU** es suficiente. Las figuras se guardan en `./images/` dentro del entorno efímero de Colab; para conservarlas, monta Google Drive o descárgalas con la pestaña *Files*.

---

## Dependencias principales

| Librería | Versión | Uso |
|---|---|---|
| pandas | 2.2.3 | Manipulación de datos |
| numpy | 2.0.2 | Operaciones numéricas |
| scikit-learn | 1.6.1 | Modelos, CV, métricas |
| xgboost | 3.0.0 | Gradient boosting |
| fairlearn | 0.12.0 | Mitigación de sesgo y métricas de equidad |
| shap | 0.46.0 | Interpretabilidad |
| matplotlib | 3.10.0 | Gráficos |
| seaborn | 0.13.2 | Gráficos estadísticos |

Lista completa y versiones exactas en `requirements.txt`.

---

## Resolución de problemas

- **`ModuleNotFoundError: fairlearn`** — instala con `pip install fairlearn==0.12.0` (problema encontrado en Colab).
- **`xgboost` falla al importar en macOS** — instala primero `libomp`: `brew install libomp`.
