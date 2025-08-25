# DiseasePrediction
The project automatically builds, trains, evaluates, and saves three separate disease prediction models.  In the end, you have:  Ready-to-use models (for Parkinson’s, Kidney, Liver).  Reports &amp; plots to see how well each model performs.

It takes three different medical datasets (Parkinson’s, Kidney Disease, and Liver Disease).

For each disease:
It cleans the raw data (fixes labels, removes unnecessary columns, handles missing values).
It builds a machine learning pipeline (preprocessing + model).
It trains a model (Logistic Regression, Random Forest, or Gradient Boosting depending on dataset).
It evaluates the model using metrics like accuracy, precision, recall, F1, ROC-AUC.
It saves the trained model (so you can load it later for predictions).
It generates reports and plots (confusion matrix, cross-validation scores, metrics in JSON).
