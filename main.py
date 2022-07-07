from flask import Flask, request, jsonify
from flask_cors import CORS
from model import clustering_model

app = Flask(__name__)
CORS(app)

@app.route('/')
def index():
    return "plusOne :D!"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json() 
    clusters, labels = clustering_model(data["inputs"])
    output = {
        "clusters": clusters,
        "labels": labels
    }
    return output
