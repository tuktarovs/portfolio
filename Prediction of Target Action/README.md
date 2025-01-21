# Classification Project

**First Major Work in Data Science**

**Task**: To predict a target action based on client behavior and data, such as purchasing a product or subscribing to a service.

**Target Metrics**:
- **ROC_AUC** > 0.65
- **Average time** from API request submission to receiving a response < 3 seconds

## Project Stages:
- **Data Analysis**: A detailed analysis of datasets was conducted in the Jupyter Notebook `Data_analys`, identifying key characteristics of clients and their behavior. Graphs depicting dependencies and correlations were provided.

- **Model Search**: In the Jupyter Notebook `Model`, feature engineering was conducted according to the results of the analysis: duplicates were removed, missing values and outliers were handled, new features were added, and the data was standardized and encoded. The search for the optimal classification model and its optimization using machine learning methods followed.

- **Distances**: The Jupyter Notebook `Distance` examined the distances between cities and Moscow, which could influence client behavior.

- **Pipeline**: In the script `pipeline.py`, a model pipeline was created and trained, including preprocessing steps, to simplify the prediction process.

**Libraries Used**:
- Pandas, Matplotlib, Seaborn, Missingno, Scikit-learn, Feature_engine, Geopy, Dill.
- Classification models: as a result of experiments, the best model turned out to be LGBM (ROC_AUC > 0.7).
- Hyperparameter tuning was performed using Bayesian optimization (package `bayes_opt`).
- The model was packaged in a Pipeline and saved using the `dill` module.

**Additional Features**:
- The model can be accessed through FastAPI. A detailed guide is provided in the presentation.

For more detailed information, please refer to the corresponding Jupyter Notebook. The conclusions and results of the work can be seen in the presentation.
