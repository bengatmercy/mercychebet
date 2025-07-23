# SYRIA  CUSTOMER CHURN PREDICTION 
Author: Mercy chebet

This project focuses on predicting customer churn using a Decision Tree classifier. The workflow includes preprocessing, model building, hyperparameter tuning, model evaluation, and final insights using performance metrics such as ROC AUC.

##Project Overview
The primary goal of this project is to build a predictive model that can accurately classify customers as likely to churn or not, based on provided features. The decision tree algorithm was selected for its interpretability and ability to capture non-linear relationships.

##Tools & Technologies
Python 3.x
Scikit-learn
Pandas
Matplotlib & Seaborn
Imbalanced-learn (for handling class imbalance)
Jupyter Notebook
##Methodology
1. Data Preparation

Imported and explored the customer dataset.
Handled missing values and performed feature encoding.
Split data into training and test sets.
2. Handling Class Imbalance

Used resampling techniques like SMOTE or RandomOverSampler to address imbalance in churn classes.
3. Pipeline Creation

Built a pipeline combining preprocessing steps and a Decision Tree Classifier.
4. Hyperparameter Tuning

Tuned the model using GridSearchCV with cross-validation to optimize the following parameters:
criterion: gini or entropy
max_depth
min_samples_split
min_samples_leaf
max_features
5. Model Evaluation

Evaluated model performance using:
Confusion Matrix
Recall, Precision, Accuracy
ROC Curve and AUC Score
ROC AUC of the final model: 0.65
6. Feature Importance

Attempted to visualize feature importances (may not apply to all pipeline configurations).

## Results
Best Parameters Found: GridSearchCV returned the optimal parameters for the Decision Tree model.
Performance: The ROC AUC of 0.65 indicates the model has moderate ability to distinguish churners from non-churners.
Overfitting Noted: The model performed better on training than on test data, suggesting some overfitting.
## Challenges & Future Work
Model Overfitting: Future iterations can incorporate pruning or ensemble methods like Random Forests.
ROC AUC Score: A score of 0.65 suggests room for performance improvement.
Feature Engineering: Creating new features or reducing irrelevant ones might improve model accuracy.

├── notebooks/
│   └── decision_tree_churn.ipynb
├── images/
│   └── roc_curve.png
├── models/
│   └── decision_tree_model.pkl
├── README.md
