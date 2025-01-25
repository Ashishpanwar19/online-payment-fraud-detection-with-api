from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)

# Load the trained model
model = pickle.load(open("C:/Users/ashis/Downloads/model.pkl","rb"))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get input values from the form
    transaction_type = request.form['type']
    amount = float(request.form['amount'])
    oldbalanceOrg = float(request.form['oldbalanceOrg'])
    newbalanceOrig = float(request.form['newbalanceOrig'])

    # Map transaction type to numerical value
    type_mapping = {
        "CASH_OUT": 1,
        "PAYMENT": 2,
        "CASH_IN": 3,
        "TRANSFER": 4,
        "DEBIT": 5
    }
    val = type_mapping.get(transaction_type, 0)  # Default to 0 if type is unknown

    # Create input array for prediction
    input_array = np.array([[val, amount, oldbalanceOrg, newbalanceOrig]])

    # Make prediction using the loaded model
    prediction = model.predict(input_array)

    # Extract the predicted output value
    output = prediction[0] if hasattr(prediction, '__iter__') else prediction

    return render_template('index.html', prediction=output)

if __name__ == '__main__':
    app.run(debug=True)
