from flask import Flask, request, jsonify
from flask_cors import CORS
from model import clustering_model
import json
import ast


app = Flask(__name__)
CORS(app)

@app.route('/')
def index():
    return "plusOne :D!"

@app.route('/predict', methods=['GET'])
def predict():
    inputs = request.args['inputs']
    print(type(inputs))
    inputs = ast.literal_eval(inputs)
    print(  type(inputs))

    print(inputs)


    clusters, labels = clustering_model(inputs)
    output = {
        "clusters": clusters,
        "labels": labels
    }
    return output
