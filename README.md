Development of an Online Payment Fraud Detection System and Prediction App

The dataset used for training and testing the model contains online transaction data. It includes the following columns:
step: Represents a unit of time where 1 step equals 1 hour. type: Type of online transaction. amount: The amount of the transaction. nameOrig: Customer starting the transaction. oldbalanceOrg: Balance before the transaction. newbalanceOrig: Balance after the transaction. nameDest: Recipient of the transaction. oldbalanceDest: Initial balance of recipient before the transaction. newbalanceDest: The new balance of recipient after the transaction. isFraud: Indicates whether the transaction is fraudulent (1) or not (0).

Model
The model uses several machine learning algorithms, including Support Vector Machine (SVM), Logistic Regression, and XGBoost, to classify transactions as fraudulent or non-fraudulent. To handle the imbalanced nature of the dataset, Synthetic Minority Over-sampling Technique (SMOTE) is used.
The models achieved high accuracy, precision, and recall scores on the test

Usage
To use the app, select a model (SVM, Logistic Regression, or XGBoost) and click "Generate and Run" to generate inference data and predict whether the transaction is fraudulent or legitimate.
Data Visualisation
The app includes data visualisation features to explore the dataset and model performance, including a correlation matrix, fraud vs. flagged fraud plot, fraudulent transactions by type, and transaction types plot.
