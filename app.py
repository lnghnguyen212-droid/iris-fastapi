from flask import Flask, render_template, request, jsonify
from predict_service import predict_iris  # Import hàm đã tách riêng

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    kernel = request.form.get('kernel', 'linear')
    result = predict_iris(kernel)  # Gọi logic từ file riêng
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)
