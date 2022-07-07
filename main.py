from flask import Flask, request, jsonify
from model import clustering_model
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

@app.route('/')
def index():
    return "plusOne :D!"

@app.route('/predict', methods=['GET','POST'])
def predict():
    data = request.get_json() 
    clusters, labels = clustering_model(data["inputs"])
    output = {
        "clusters": clusters,
        "labels": labels
    }
    return output
