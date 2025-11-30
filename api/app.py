from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict_endpoint():
    data = request.get_json()
    # prediction = predict(model, data)
    return jsonify({'prediction': 'dummy_value'})

if __name__ == '__main__':
    app.run(debug=True)
