# Using flask to make an api
# import necessary libraries and functions
from flask import Flask, jsonify, request
import joblib
import traceback
import json
import os
import numpy as np
  
# creating a Flask app
app = Flask(__name__)

__locations = None
__data_columns = None

file_dir = os.path.dirname(os.path.realpath('__file__'))

model_path = os.path.join(file_dir, 'model/house-price-model.joblib')
model = joblib.load(model_path)

json_path = os.path.join(file_dir, 'model/columns.json')
with open(json_path) as f:
    __data_columns = json.load(f)["data_columns"]
    __locations = __data_columns[3:]

# returns the predict that we send when we use the model in POST method.
@app.route('/get_locations_names', methods = ['GET'])
def getLocationName():
    try:
        response = jsonify(__locations)
        return response
    except:
        return jsonify(trace = traceback.format_exc())

# returns the predict that we send when we use the model in POST method.
@app.route('/get_estimated_price', methods = ['POST'])
def getEstimatedPrice():
    if model:
        try:
            data = request.json
            location = data.get('location')
            sqft = data.get('sqft')
            bhk = data.get('bhk')
            bath = data.get('bath')
            
            loc_index = __data_columns.index(location.lower())
    
            x = np.zeros(len(__data_columns))
            x[0] = sqft
            x[1] = bath
            x[1] = bhk
            
            if loc_index >= 0:
                x[loc_index] = 1
            
            price = round(model.predict([x])[0],2)
            return jsonify({ "estimatedPrice": price })
        except:
            return jsonify(trace = traceback.format_exc())
  
# driver function
if __name__ == '__main__':
    app.run(port=8000, debug = True)