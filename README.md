# Bank Account Fraud Detection with Machine Learning

Proyecto final de Aprendizaje Automático centrado en la detección de fraude en apertura de cuentas bancarias mediante técnicas de Machine Learning sobre datos tabulares.

El trabajo utiliza la suite de datasets **Bank Account Fraud (BAF)** propuesta por Jesus et al. en NeurIPS 2022, diseñada específicamente para evaluar modelos bajo escenarios realistas de fraude financiero: fuerte desbalanceo de clases, atributos sensibles, dinámica temporal y posibles problemas de fairness.

---

# Objetivos del proyecto

Los principales objetivos del proyecto son:

- Analizar y preprocesar el dataset BAF.
- Construir modelos de clasificación para detección de fraude.
- Evaluar el impacto del desbalanceo de clases.
- Analizar estabilidad temporal y generalización.
- Estudiar métricas de fairness y comportamiento por grupos sensibles.
- Comparar distintos modelos y estrategias de entrenamiento.

---

# Dataset

Dataset utilizado:

- **Bank Account Fraud (BAF) Dataset Suite**
- Paper original:
  - *Turning the Tables: Biased, Imbalanced, Dynamic Tabular Datasets for ML Evaluation*
- NeurIPS 2022

Enlace oficial:

- https://github.com/feedzai/bank-account-fraud
- https://www.kaggle.com/datasets/sgpjesus/bank-account-fraud-dataset-neurips-2022

---

# Estructura del proyecto

```bash
.
├── data/                   # Datos del proyecto
├── notebooks/              # Notebooks de experimentación
├── src/                    # Código fuente auxiliar
├── models/                 # Modelos entrenados
├── figures/                # Figuras y gráficas
├── requirements.txt        # Dependencias
├── README.md
└── informe/                # Memoria en LaTeX
