from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import joblib

# Nạp mô hình SVM đã huấn luyện
model = joblib.load("svm_model.pkl")

app = FastAPI(title="Iris Classification Web App")

class IrisInput(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

species = {0: "Setosa 🌸", 1: "Versicolor 🌺", 2: "Virginica 🌻"}

# 1. Trang web giao diện người dùng (HTML/CSS)
@app.get("/", response_class=HTMLResponse)
def home_ui():
    html_content = """
    <!DOCTYPE html>
    <html lang="vi">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Dự đoán loài hoa Iris - Mô hình SVM</title>
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
        <style>
            body { background-color: #f4f7f6; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; }
            .card { border-radius: 15px; box-shadow: 0 10px 25px rgba(0,0,0,0.1); }
            .btn-custom { background: linear-gradient(135deg, #6a11cb 0%, #2575fc 100%); color: white; border: none; }
            .btn-custom:hover { opacity: 0.9; color: white; }
            .result-box { display: none; margin-top: 20px; padding: 15px; border-radius: 10px; font-weight: bold; text-align: center; }
        </style>
    </head>
    <body>
        <div class="container py-5">
            <div class="row justify-content-center">
                <div class="col-md-6">
                    <div class="card p-4">
                        <h3 class="text-center text-primary mb-3">🌸 Phân Loại Hoa Iris (SVM)</h3>
                        <p class="text-muted text-center">Nhập kích thước các lá đài và cánh hoa để dự đoán loài hoa</p>
                        
                        <form id="irisForm">
                            <div class="mb-3">
                                <label class="form-label">Chiều dài lá đài (Sepal Length): <b id="val_sl">5.1</b> cm</label>
                                <input type="range" class="form-range" id="sepal_length" min="4.0" max="8.0" step="0.1" value="5.1" oninput="document.getElementById('val_sl').innerText=this.value">
                            </div>
                            <div class="mb-3">
                                <label class="form-label">Chiều rộng lá đài (Sepal Width): <b id="val_sw">3.5</b> cm</label>
                                <input type="range" class="form-range" id="sepal_width" min="2.0" max="5.0" step="0.1" value="3.5" oninput="document.getElementById('val_sw').innerText=this.value">
                            </div>
                            <div class="mb-3">
                                <label class="form-label">Chiều dài cánh hoa (Petal Length): <b id="val_pl">1.4</b> cm</label>
                                <input type="range" class="form-range" id="petal_length" min="1.0" max="7.0" step="0.1" value="1.4" oninput="document.getElementById('val_pl').innerText=this.value">
                            </div>
                            <div class="mb-3">
                                <label class="form-label">Chiều rộng cánh hoa (Petal Width): <b id="val_pw">0.2</b> cm</label>
                                <input type="range" class="form-range" id="petal_width" min="0.1" max="2.5" step="0.1" value="0.2" oninput="document.getElementById('val_pw').innerText=this.value">
                            </div>
                            
                            <button type="button" class="btn btn-custom w-100 py-2 fs-5" onclick="makePrediction()">Dự đoán loài hoa</button>
                        </form>

                        <div id="resultBox" class="result-box alert alert-success">
                            Kết quả dự đoán: <span id="predictionResult" class="fs-4 text-uppercase"></span>
                        </div>
                    </div>
                </div>
            </div>
        </div>

        <script>
            async function makePrediction() {
                const data = {
                    sepal_length: parseFloat(document.getElementById('sepal_length').value),
                    sepal_width: parseFloat(document.getElementById('sepal_width').value),
                    petal_length: parseFloat(document.getElementById('petal_length').value),
                    petal_width: parseFloat(document.getElementById('petal_width').value)
                };

                const response = await fetch('/predict', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(data)
                });

                const result = await response.json();
                const resultBox = document.getElementById('resultBox');
                const resultText = document.getElementById('predictionResult');

                resultText.innerText = result.prediction;
                resultBox.style.display = 'block';
            }
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

# 2. API Cổng dự đoán
@app.post("/predict")
def predict(data: IrisInput):
    features = [[data.sepal_length, data.sepal_width, data.petal_length, data.petal_width]]
    prediction = int(model.predict(features)[0])
    return {"class_id": prediction, "prediction": species[prediction]}
