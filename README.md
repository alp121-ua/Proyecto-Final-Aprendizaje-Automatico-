# Bank Account Fraud Detection with Machine Learning

Proyecto final de la asignatura **Aprendizaje Avanzado** del Grado en Ingeniería en Inteligencia Artificial de la Universidad de Alicante.

El objetivo del proyecto es estudiar el problema de **detección automática de fraude en solicitudes de apertura de cuentas bancarias** utilizando técnicas de aprendizaje automático supervisado sobre datos tabulares. El trabajo se centra no solo en maximizar el rendimiento predictivo, sino también en analizar problemáticas propias de sistemas de Machine Learning aplicados a escenarios reales, como el **desbalanceo de clases**, la **robustez ante cambios de distribución**, la **fairness**, la **interpretabilidad** y el **apoyo a la decisión humana**.

## Dataset utilizado

El proyecto utiliza la suite de datasets **Bank Account Fraud (BAF)**, propuesta en el trabajo:

> *Turning the Tables: Biased, Imbalanced, Dynamic Tabular Datasets for ML Evaluation*  
> NeurIPS 2022 Datasets and Benchmarks Track

BAF es un conjunto de datos tabular sintético generado a partir de datos reales anonimizados de apertura de cuentas bancarias. Cada instancia representa una solicitud de apertura de cuenta y la variable objetivo es `fraud_bool`, que indica si la solicitud es fraudulenta (`1`) o legítima (`0`).

El dataset resulta especialmente adecuado para este proyecto porque reproduce varias dificultades habituales en problemas reales de fraude financiero:

- Fuerte desbalanceo de clases.
- Variables numéricas, categóricas y binarias.
- Atributos potencialmente sensibles.
- Información temporal mediante la variable `month`.
- Variantes diseñadas para evaluar robustez, sesgo y cambios de distribución.

Enlaces oficiales:

- GitHub: https://github.com/feedzai/bank-account-fraud
- Kaggle: https://www.kaggle.com/datasets/sgpjesus/bank-account-fraud-dataset-neurips-2022

## Objetivos del proyecto

Los principales objetivos del trabajo son:

- Formular el problema como una tarea de clasificación binaria supervisada.
- Realizar un análisis exploratorio del dataset BAF.
- Identificar las principales problemáticas del dominio: desbalanceo, robustez, fairness e interpretabilidad.
- Comparar distintos modelos de clasificación sobre datos tabulares.
- Evaluar estrategias de tratamiento del desbalanceo, incluyendo ponderación de clases y técnicas de sobremuestreo.
- Optimizar los modelos más prometedores mediante búsqueda de hiperparámetros.
- Evaluar la robustez del modelo final sobre las variantes del dataset BAF.
- Analizar la importancia de variables y generar explicaciones contrafactuales como apoyo a la decisión.

## Metodología

El trabajo sigue un pipeline experimental progresivo:

1. **Análisis exploratorio de datos (EDA)**  
   Se analiza la distribución de clases, las variables numéricas y categóricas, las correlaciones, los outliers y las primeras señales predictivas del dataset.

2. **Preprocesamiento**  
   Se codifican las variables categóricas, se prepara el dataset para el entrenamiento y se mantiene una partición estratificada entre entrenamiento y test.  
   Aunque se estudiaron posibles tratamientos de outliers, finalmente se decidió conservarlos, ya que en problemas de fraude pueden representar precisamente comportamientos anómalos relevantes.

3. **Criba inicial de modelos**  
   Se comparan distintos algoritmos de clasificación, incluyendo modelos lineales, árboles, ensembles y métodos de gradient boosting.

4. **Evaluación de estrategias de muestreo**  
   Se prueban técnicas como SMOTE, BorderlineSMOTE y ADASYN. Los resultados muestran que el mejor rendimiento se obtiene sin aplicar sobremuestreo, utilizando CatBoost con ponderación de clases.

5. **Optimización de hiperparámetros**  
   Los mejores modelos de la criba inicial se optimizan mediante Optuna. El modelo seleccionado finalmente es CatBoost.

6. **Evaluación final**  
   El modelo final se entrena sobre el conjunto completo de entrenamiento y se evalúa sobre el dataset Base y las variantes BAF.

7. **Interpretabilidad y apoyo a la decisión**  
   Se analizan las variables más importantes del modelo y se generan explicaciones contrafactuales mediante DiCE para estudiar qué factores influyen en las predicciones.

## Modelos evaluados

Durante el proyecto se compararon diferentes modelos de clasificación:

- LinearSVC
- Decision Tree
- Random Forest
- AdaBoost
- Gradient Boosting
- HistGradientBoosting
- XGBoost
- LightGBM
- CatBoost

Tras la comparación experimental y la optimización de hiperparámetros, **CatBoost** fue seleccionado como modelo final por ofrecer el mejor equilibrio entre rendimiento, robustez y coste computacional.

## Métricas de evaluación

Debido al fuerte desbalanceo de clases del dataset, la métrica principal utilizada es **Average Precision (AP)**, derivada de la curva Precision-Recall.

También se reportan métricas complementarias:

- ROC-AUC
- F1-score
- Recall
- Precision

La accuracy no se utiliza como métrica principal, ya que en este problema puede resultar engañosa: un modelo que predijera siempre la clase mayoritaria obtendría una precisión global muy alta, pero no detectaría fraude.

## Resultados principales

El modelo final CatBoost, entrenado sin sobremuestreo y con ponderación de clases, obtuvo sobre el dataset Base:

| Métrica | Valor |
|---|---:|
| Average Precision | 0.1842 |
| ROC-AUC | 0.8991 |
| F1-score | 0.2494 |
| Recall | 0.2684 |
| Precision | 0.2330 |

Estos resultados deben interpretarse en el contexto de un problema altamente desbalanceado, donde solo alrededor del 1.1% de las solicitudes corresponden a fraude.

El modelo no debe entenderse como un sistema autónomo para aceptar o rechazar solicitudes, sino como una herramienta de apoyo para priorizar revisiones de riesgo por parte de equipos de seguridad bancaria.

## Conclusiones

El proyecto confirma que la detección de fraude financiero es un problema complejo, sensible y dinámico. El fuerte desbalanceo de clases condiciona tanto la elección de métricas como el diseño del modelo. Además, los resultados muestran que técnicas clásicas de sobremuestreo no siempre mejoran el rendimiento en este dominio, ya que pueden introducir patrones artificiales poco representativos.

CatBoost fue el modelo con mejor comportamiento global en los experimentos realizados. Sin embargo, la evaluación sobre variantes del dataset BAF mostró que el rendimiento disminuye ante cambios en la distribución de los datos, lo que refuerza la necesidad de monitorización, recalibración y reentrenamiento periódico en un posible entorno real.

Finalmente, el análisis de importancia de variables y los contrafactuales permiten aportar una capa de interpretabilidad útil para entender las decisiones del modelo y apoyar el trabajo de analistas humanos.

## Requisitos

Las principales dependencias utilizadas en el proyecto son:

- Python
- NumPy
- pandas
- scikit-learn
- imbalanced-learn
- CatBoost
- XGBoost
- LightGBM
- Optuna
- matplotlib
- seaborn
- SHAP
- DiCE-ML

Para instalar las dependencias:

```bash
pip install -r requirements.txt

```
# Estructura del proyecto

```bash
.
├── eda/                 # Preprocesamiento dataset
├── img/                # Figuras y gráficas
├── procesamiento_modelos/  # Procesamiento completo 
├── requirements.txt        # Dependencias
├── README.md
└── informe/                # Memoria en LaTeX

```