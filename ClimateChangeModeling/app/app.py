from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/')
def index():
    return 'Welcome to the Climate Change Modeling API!'

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    # Add model prediction logic here
    return jsonify({'prediction': 'Mock result'})

if __name__ == '__main__':
    app.run(debug=True)
