from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import joblib

# Nạp mô hình SVM đã huấn luyện
model = joblib.load("svm_model.pkl")

app = FastAPI(title="Iris Pro Classification Web App")

class IrisInput(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

# Thông tin chi tiết + Link ảnh thực tế từng loài hoa
species_info = {
    0: {
        "name": "Iris Setosa 🌸",
        "img": "https://upload.wikimedia.org/wikipedia/commons/5/56/Kosaciec_szczecinkowaty_Iris_setosa.jpg",
        "desc": "Lá đài rộng, cánh hoa ngắn. Loài hoa này rất dễ phân biệt với các loài khác."
    },
    1: {
        "name": "Iris Versicolor 🌺",
        "img": "https://upload.wikimedia.org/wikipedia/commons/4/41/Iris_versicolor_3.jpg",
        "desc": "Kích thước trung bình, màu sắc sặc sỡ từ xanh dừa đến tím đậm."
    },
    2: {
        "name": "Iris Virginica 🌻",
        "img": "https://upload.wikimedia.org/wikipedia/commons/9/9f/Iris_virginica.jpg",
        "desc": "Cánh hoa và lá đài lớn nhất trong 3 loài, màu tím nhạt quyến rũ."
    }
}

@app.get("/", response_class=HTMLResponse)
def home_ui():
    html_content = """
    <!DOCTYPE html>
    <html lang="vi">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Hệ Thống Phân Loại Hoa Iris Thông Minh - SVM</title>
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
        <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
        <style>
            body { background: linear-gradient(135deg, #e0c3fc 0%, #8ec5fc 100%); min-height: 100vh; font-family: 'Segoe UI', sans-serif; }
            .card-main { border-radius: 20px; border: none; box-shadow: 0 15px 35px rgba(0,0,0,0.15); background: rgba(255, 255, 255, 0.95); backdrop-filter: blur(10px); }
            .btn-predict { background: linear-gradient(45deg, #11998e, #38ef7d); color: white; font-weight: bold; border: none; border-radius: 10px; transition: all 0.3s ease; }
            .btn-predict:hover { transform: translateY(-2px); box-shadow: 0 5px 15px rgba(56, 239, 125, 0.4); color: white; }
            .flower-img { width: 100%; height: 220px; object-fit: cover; border-radius: 15px; box-shadow: 0 5px 15px rgba(0,0,0,0.1); }
            .result-card { display: none; animation: fadeIn 0.5s ease-in-out; }
            @keyframes fadeIn { from { opacity: 0; transform: translateY(10px); } to { opacity: 1; transform: translateY(0); } }
        </style>
    </head>
    <body class="py-5">
        <div class="container">
            <div class="row justify-content-center">
                <div class="col-lg-10">
                    <div class="card card-main p-4 p-md-5">
                        <h2 class="text-center fw-bold text-primary mb-2">🌸 Dự Đoán Phân Loại Hoa Iris </h2>
                        <p class="text-center text-muted mb-4">Hệ thống nhận diện loài hoa thông minh dựa trên giải thuật Support Vector Machine</p>
                        
                        <div class="row g-4">
                            <!-- Cột nhập số liệu -->
                            <div class="col-md-6 border-end">
                                <h5 class="fw-bold mb-3 text-secondary">📏 Nhập kích thước mẫu hoa:</h5>
                                <form id="irisForm">
                                    <div class="mb-3">
                                        <label class="form-label d-flex justify-content-between">
                                            <span>Chiều dài lá đài (Sepal Length)</span>
                                            <span class="badge bg-primary fs-6"><span id="val_sl">5.1</span> cm</span>
                                        </label>
                                        <input type="range" class="form-range" id="sepal_length" min="4.0" max="8.0" step="0.1" value="5.1" oninput="updateVal('val_sl', this.value)">
                                    </div>
                                    <div class="mb-3">
                                        <label class="form-label d-flex justify-content-between">
                                            <span>Chiều rộng lá đài (Sepal Width)</span>
                                            <span class="badge bg-primary fs-6"><span id="val_sw">3.5</span> cm</span>
                                        </label>
                                        <input type="range" class="form-range" id="sepal_width" min="2.0" max="5.0" step="0.1" value="3.5" oninput="updateVal('val_sw', this.value)">
                                    </div>
                                    <div class="mb-3">
                                        <label class="form-label d-flex justify-content-between">
                                            <span>Chiều dài cánh hoa (Petal Length)</span>
                                            <span class="badge bg-primary fs-6"><span id="val_pl">1.4</span> cm</span>
                                        </label>
                                        <input type="range" class="form-range" id="petal_length" min="1.0" max="7.0" step="0.1" value="1.4" oninput="updateVal('val_pl', this.value)">
                                    </div>
                                    <div class="mb-3">
                                        <label class="form-label d-flex justify-content-between">
                                            <span>Chiều rộng cánh hoa (Petal Width)</span>
                                            <span class="badge bg-primary fs-6"><span id="val_pw">0.2</span> cm</span>
                                        </label>
                                        <input type="range" class="form-range" id="petal_width" min="0.1" max="2.5" step="0.1" value="0.2" oninput="updateVal('val_pw', this.value)">
                                    </div>
                                    
                                    <button type="button" class="btn btn-predict w-100 py-3 mt-2 fs-5" onclick="makePrediction()">✨ Phân Loại Ngay</button>
                                </form>
                            </div>

                            <!-- Cột hiển thị kết quả & Ảnh & Biểu đồ -->
                            <div class="col-md-6 d-flex flex-column justify-content-center">
                                <div id="placeholderText" class="text-center text-muted my-auto">
                                    <img src="https://cdn-icons-png.flaticon.com/512/620/620851.png" style="width: 100px; opacity: 0.5;" class="mb-3">
                                    <h5>Kéo chọn các chỉ số và bấm nút để xem AI dự đoán!</h5>
                                </div>

                                <div id="resultCard" class="result-card">
                                    <div class="text-center mb-3">
                                        <span class="badge bg-success px-3 py-2 fs-6">Kết quả nhận diện:</span>
                                        <h3 id="flowerName" class="fw-bold text-success mt-2"></h3>
                                    </div>

                                    <img id="flowerImg" src="" class="flower-img mb-3" alt="Flower Image">
                                    <p id="flowerDesc" class="text-muted small text-center mb-3"></p>

                                    <!-- Biểu đồ Radar -->
                                    <div style="height: 200px;">
                                        <canvas id="radarChart"></canvas>
                                    </div>
                                </div>
                            </div>
                        </div>

                    </div>
                </div>
            </div>
        </div>

        <script>
            let radarChart = null;

            function updateVal(id, val) {
                document.getElementById(id).innerText = val;
            }

            async function makePrediction() {
                const sl = parseFloat(document.getElementById('sepal_length').value);
                const sw = parseFloat(document.getElementById('sepal_width').value);
                const pl = parseFloat(document.getElementById('petal_length').value);
                const pw = parseFloat(document.getElementById('petal_width').value);

                const response = await fetch('/predict', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ sepal_length: sl, sepal_width: sw, petal_length: pl, petal_width: pw })
                });

                const result = await response.json();

                // Ẩn placeholder, hiện kết quả
                document.getElementById('placeholderText').style.display = 'none';
                document.getElementById('resultCard').style.display = 'block';

                document.getElementById('flowerName').innerText = result.info.name;
                document.getElementById('flowerImg').src = result.info.img;
                document.getElementById('flowerDesc').innerText = result.info.desc;

                // Vẽ/Cập nhật Biểu đồ Radar
                renderRadar([sl, sw, pl, pw]);
            }

            function renderRadar(inputData) {
                const ctx = document.getElementById('radarChart').getContext('2d');
                
                if (radarChart) { radarChart.destroy(); }

                radarChart = new Chart(ctx, {
                    type: 'radar',
                    data: {
                        labels: ['Sepal Length', 'Sepal Width', 'Petal Length', 'Petal Width'],
                        datasets: [{
                            label: 'Chỉ số nhập vào',
                            data: inputData,
                            backgroundColor: 'rgba(56, 239, 125, 0.2)',
                            borderColor: '#38ef7d',
                            pointBackgroundColor: '#11998e'
                        }]
                    },
                    options: {
                        responsive: true,
                        maintainAspectRatio: false,
                        scales: { r: { min: 0, max: 8 } }
                    }
                });
            }
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

@app.post("/predict")
def predict(data: IrisInput):
    features = [[data.sepal_length, data.sepal_width, data.petal_length, data.petal_width]]
    prediction = int(model.predict(features)[0])
    
    return {
        "class_id": prediction,
        "prediction": species_info[prediction]["name"],
        "info": species_info[prediction]
    }
