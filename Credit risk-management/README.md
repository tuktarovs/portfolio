#Credit Risk Management

**Predicting Borrower Default on Loans**

**Project Goal**

Develop a machine learning model to predict the probability of borrower default.

**Key Metrics**

**ROC AUC ≥ 0.75** (achieved **0.77**).

**Dataset**
-Size: 26.1 million records.
-Features: 61 (20 binary, 30 categorical).
-Class imbalance: Defaulted borrowers make up 3.5% of the dataset.

**Project Stages**
1. Data Analysis
Merging and preprocessing datasets.
Exploratory analysis was complicated due to the lack of semantic meaning in some features.
2. Feature Engineering
Aggregation of data by borrower ID.
Encoding of categorical features.
Class imbalance handling methods (downsampling, upsampling) did not improve performance.
3. Model Training and Optimization
Various classification models were tested.
Best model: LGBM (LightGBM) — ROC AUC 0.77.
Hyperparameter tuning using Bayesian Optimization (bayes_opt).
Cross-validation for final model selection.
4. Model Deployment
The model was packaged into a Pipeline and saved using dill.

**Libraries Used**

Pandas, Scikit-learn, Matplotlib, Seaborn, Dill
LightGBM (gradient boosting)
Bayesian Optimization (bayes_opt)

**Results**

Achieved ROC AUC 0.77 (above the target threshold of 0.75).
Developed a reusable Pipeline for model deployment.
Detailed model analysis is available in the Jupyter Notebook.
Final insights and key takeaways are presented in a report.