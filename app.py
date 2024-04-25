from flask import Flask, jsonify, request
import pickle
import numpy as np
from flask_cors import CORS 


app = Flask(__name__)
CORS(app)

# Load the pre-trained model from the pickle file
with open('insu_rf.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route('/predict', methods=['POST'])
def predict():
    # Get the input data from the request
    data = request.get_json()
    
    # Extract values from the input data dictionary
    input_values = list(data.values())
    
    # Convert input values to a 2D array-like object
    input_array = np.array(input_values).reshape(1, -1)
    
    # Make prediction using the model
    prediction = model.predict(input_array)
    
    # Return the prediction as JSON response
    return jsonify({'prediction': prediction.tolist()})

if __name__ == '__main__':
    app.run(debug=True)
