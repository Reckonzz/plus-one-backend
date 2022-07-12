from flask import Flask, request, jsonify
from flask_cors import CORS
from model import clustering_model

app = Flask(__name__)
CORS(app)

@app.route('/')
def index():
    return "plusOne :D!"

@app.route('/predict', methods=['GET'])
def predict():
    args = request.args
    print(args)
    print(args['inputs'])
    print('end')
    clusters, labels = clustering_model(args["inputs"])
    output = {
        "clusters": clusters,
        "labels": labels
    }
    return output
