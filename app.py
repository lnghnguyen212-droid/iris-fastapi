import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

# Nạp mô hình SVM đã huấn luyện
try:
    model = joblib.load("svm_model.pkl")
except Exception as e:
    print(f"Lưu ý: Chưa tìm thấy file 'svm_model.pkl' hoặc lỗi nạp file: {e}")
    model = None

app = FastAPI(title="Iris AI Enterprise Suite Pro")


class IrisInput(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float


species_info = {
    0: {
        "name": "IRIS SETOSA",
        "badge": "Loài Đặc Hữu - Nhóm 01",
        "img": "https://upload.wikimedia.org/wikipedia/commons/5/56/Kosaciec_szczecinkowaty_Iris_setosa.jpg",
        "desc": "Đặc trưng bởi lá đài rộng, cánh hoa nhỏ gọn. Mô hình SVM nhận diện loài này với độ tin cậy cao.",
        "habitat": "Vùng khí hậu ôn đới, đầm lầy",
        "origin": "Bắc Mỹ & Đông Bắc Á",
        "probs": [100, 0, 0],
    },
    1: {
        "name": "IRIS VERSICOLOR",
        "badge": "Loài Phổ Biến - Nhóm 02",
        "img": "https://upload.wikimedia.org/wikipedia/commons/4/41/Iris_versicolor_3.jpg",
        "desc": "Kích thước trung bình, dải màu từ lam xám đến tím sẫm. Thuộc nhóm trung gian có đặc trưng biến thiên cao.",
        "habitat": "Ven sông, suối, vùng ven hồ",
        "origin": "Đông Bắc Bắc Mỹ",
        "probs": [2, 94, 4],
    },
    2: {
        "name": "IRIS VIRGINICA",
        "badge": "Loài Kích Thước Lớn - Nhóm 03",
        "img": "https://upload.wikimedia.org/wikipedia/commons/9/9f/Iris_virginica.jpg",
        "desc": "Sở hữu cánh hoa và lá đài phát triển tối đa. Cấu trúc hình học vượt trội so với hai nhóm loài còn lại.",
        "habitat": "Đồng cỏ ẩm ướt, đầm lầy cạn",
        "origin": "Đông Nam Hoa Kỳ",
        "probs": [0, 5, 95],
    },
}


# API Dự đoán cho Frontend gọi qua Fetch
@app.post("/predict")
def predict(data: IrisInput):
    features = [
        [
            data.sepal_length,
            data.sepal_width,
            data.petal_length,
            data.petal_width,
        ]
    ]

    if model is not None:
        try:
            pred_class = int(model.predict(features)[0])
            # Tính xác suất nếu mô hình hỗ trợ predict_proba
            if hasattr(model, "predict_proba"):
                probabilities = (
                    (model.predict_proba(features)[0] * 100)
                    .round(1)
                    .tolist()
                )
            else:
                probabilities = species_info[pred_class]["probs"]
        except Exception as e:
            raise HTTPException(
                status_code=500, detail=f"Lỗi khi dự đoán: {e}"
            )
    else:
        # Giả lập logic heuristic nếu chưa nạp mô hình .pkl
        if data.petal_length < 2.5:
            pred_class = 0
            probabilities = [98.5, 1.0, 0.5]
        elif data.petal_length < 4.8:
            pred_class = 1
            probabilities = [1.2, 92.3, 6.5]
        else:
            pred_class = 2
            probabilities = [0.1, 4.4, 95.5]

    info = species_info[pred_class].copy()
    info["calculated_probs"] = probabilities
    info["pred_class"] = pred_class
    return info


@app.get("/", response_class=HTMLResponse)
def home_ui():
    html_content = """
    <!DOCTYPE html>
    <html lang="vi" data-bs-theme="dark">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Iris Analytics Pro</title>
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
        <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;600;700;800&display=swap" rel="stylesheet">
        <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
        <style>
            :root {
                --bg-main: #0a0d18;
                --card-bg: rgba(23, 31, 56, 0.8);
                --card-border: rgba(255, 255, 255, 0.12);
                --accent-cyan: #00f2fe;
                --accent-pink: #ff007f;
                --accent-purple: #7928ca;
                --text-main: #f8fafc;
                --text-sub: #cbd5e1;
                --stat-bg: rgba(0, 0, 0, 0.3);
            }

            [data-bs-theme="light"] {
                --bg-main: #f1f5f9;
                --card-bg: #ffffff;
                --card-border: #e2e8f0;
                --accent-cyan: #0284c7;
                --accent-pink: #e11d48;
                --accent-purple: #6b21a8;
                --text-main: #0f172a;
                --text-sub: #475569;
                --stat-bg: #f8fafc;
            }

            body {
                background-color: var(--bg-main);
                color: var(--text-main);
                font-family: 'Plus Jakarta Sans', sans-serif;
                min-height: 100vh;
                transition: background-color 0.3s, color 0.3s;
            }

            .navbar-custom {
                background: var(--card-bg);
                backdrop-filter: blur(16px);
                border-bottom: 1px solid var(--card-border);
            }

            .glass-card {
                background: var(--card-bg);
                border: 1px solid var(--card-border);
                backdrop-filter: blur(20px);
                border-radius: 20px;
                box-shadow: 0 10px 30px rgba(0,0,0,0.1);
            }

            .nav-pills .nav-link {
                color: var(--text-sub);
                border-radius: 12px;
                padding: 10px 24px;
                font-weight: 700;
            }
            .nav-pills .nav-link.active {
                background: linear-gradient(135deg, var(--accent-purple) 0%, var(--accent-pink) 100%);
                color: #fff;
            }

            .form-label-custom {
                font-size: 0.95rem;
                font-weight: 700;
                color: var(--text-main);
                display: flex;
                justify-content: space-between;
            }

            .custom-input {
                background: var(--stat-bg) !important;
                border: 2px solid var(--card-border) !important;
                color: var(--accent-cyan) !important;
                font-weight: 800;
                font-size: 1.1rem;
                text-align: center;
                border-radius: 10px;
            }

            .btn-step {
                background: var(--stat-bg);
                border: 1px solid var(--card-border);
                color: var(--text-main);
                font-weight: 800;
                width: 42px;
                border-radius: 10px !important;
            }

            .btn-analyze {
                background: linear-gradient(135deg, var(--accent-cyan) 0%, #3b82f6 100%);
                color: #fff;
                font-weight: 800;
                font-size: 1.1rem;
                border: none;
                border-radius: 14px;
            }

            .img-box-large {
                border-radius: 16px;
                overflow: hidden;
                border: 2px solid var(--card-border);
                background: #000;
                height: 260px;
                width: 100%;
            }
            .img-box-large img {
                width: 100%;
                height: 100%;
                object-fit: cover;
            }

            .stat-box {
                background: var(--stat-bg);
                border: 1px solid var(--card-border);
                border-radius: 14px;
                padding: 12px 16px;
            }

            .progress-custom {
                height: 10px;
                background: rgba(0, 0, 0, 0.2);
                border-radius: 10px;
            }

            .flower-title {
                font-size: 2rem;
                font-weight: 800;
                color: var(--accent-cyan);
            }
        </style>
    </head>
    <body>

        <!-- Header Navigation -->
        <nav class="navbar navbar-expand-lg navbar-custom sticky-top px-4">
            <div class="container-fluid">
                <a class="navbar-brand d-flex align-items-center gap-2 fw-800 fs-4" href="#">
                    <span>✨</span> IRIS ANALYTICS PRO
                </a>
                <div class="d-flex align-items-center gap-3">
                    <button class="btn btn-outline-primary btn-sm rounded-pill px-3 fw-700" onclick="toggleTheme()">🌓 Đổi Theme</button>
                    <span class="badge bg-success p-2 fs-6">SVM Engine Active</span>
                </div>
            </div>
        </nav>

        <div class="container-fluid px-4 py-3">
            <!-- Navigation Tabs -->
            <ul class="nav nav-pills mb-3 justify-content-center gap-2" id="mainTabs">
                <li class="nav-item">
                    <button class="nav-link active" id="predict-tab" data-bs-toggle="pill" data-bs-target="#tab-predict">🔮 Phân Loại AI</button>
                </li>
                <li class="nav-item">
                    <button class="nav-link" id="library-tab" data-bs-toggle="pill" data-bs-target="#tab-library">📚 Thư Viện Sinh Học</button>
                </li>
            </ul>

            <div class="tab-content">
                <!-- TAB 1: PREDICT ENGINE -->
                <div class="tab-pane fade show active" id="tab-predict">
                    <div class="row g-3">
                        <!-- Input Controls -->
                        <div class="col-xl-4 col-lg-5">
                            <div class="glass-card p-4 h-100">
                                <h5 class="fw-800 mb-3 text-warning">🎛️ Bảng Điều Khiển Thông Số</h5>
                                
                                <div class="mb-3">
                                    <label class="form-label text-sub fw-600 small mb-1">Mẫu thử nhanh:</label>
                                    <div class="d-flex gap-2">
                                        <button class="btn btn-sm btn-outline-info flex-fill fw-700 py-2" onclick="loadPreset(5.1, 3.5, 1.4, 0.2)">Setosa</button>
                                        <button class="btn btn-sm btn-outline-warning flex-fill fw-700 py-2" onclick="loadPreset(6.0, 2.9, 4.5, 1.5)">Versicolor</button>
                                        <button class="btn btn-sm btn-outline-danger flex-fill fw-700 py-2" onclick="loadPreset(6.5, 3.0, 5.5, 2.0)">Virginica</button>
                                    </div>
                                </div>

                                <hr class="border-secondary opacity-25">

                                <!-- Inputs -->
                                <div class="mb-3">
                                    <div class="form-label-custom mb-1">
                                        <span>Sepal Length (Dài đài)</span>
                                        <span class="text-info fw-800" id="val_sl">5.1 cm</span>
                                    </div>
                                    <div class="input-group mb-1">
                                        <button class="btn btn-step" onclick="stepVal('sepal_length', -0.1)">-</button>
                                        <input type="number" class="form-control custom-input" id="sepal_length" value="5.1" step="0.1" oninput="syncSlider('sepal_length', 'range_sl', 'val_sl')">
                                        <button class="btn btn-step" onclick="stepVal('sepal_length', 0.1)">+</button>
                                    </div>
                                    <input type="range" class="form-range" id="range_sl" min="4.0" max="8.0" step="0.1" value="5.1" oninput="syncInput('sepal_length', 'range_sl', 'val_sl')">
                                </div>

                                <div class="mb-3">
                                    <div class="form-label-custom mb-1">
                                        <span>Sepal Width (Rộng đài)</span>
                                        <span class="text-info fw-800" id="val_sw">3.5 cm</span>
                                    </div>
                                    <div class="input-group mb-1">
                                        <button class="btn btn-step" onclick="stepVal('sepal_width', -0.1)">-</button>
                                        <input type="number" class="form-control custom-input" id="sepal_width" value="3.5" step="0.1" oninput="syncSlider('sepal_width', 'range_sw', 'val_sw')">
                                        <button class="btn btn-step" onclick="stepVal('sepal_width', 0.1)">+</button>
                                    </div>
                                    <input type="range" class="form-range" id="range_sw" min="2.0" max="4.5" step="0.1" value="3.5" oninput="syncInput('sepal_width', 'range_sw', 'val_sw')">
                                </div>

                                <div class="mb-3">
                                    <div class="form-label-custom mb-1">
                                        <span>Petal Length (Dài cánh)</span>
                                        <span class="text-info fw-800" id="val_pl">1.4 cm</span>
                                    </div>
                                    <div class="input-group mb-1">
                                        <button class="btn btn-step" onclick="stepVal('petal_length', -0.1)">-</button>
                                        <input type="number" class="form-control custom-input" id="petal_length" value="1.4" step="0.1" oninput="syncSlider('petal_length', 'range_pl', 'val_pl')">
                                        <button class="btn btn-step" onclick="stepVal('petal_length', 0.1)">+</button>
                                    </div>
                                    <input type="range" class="form-range" id="range_pl" min="1.0" max="7.0" step="0.1" value="1.4" oninput="syncInput('petal_length', 'range_pl', 'val_pl')">
                                </div>

                                <div class="mb-3">
                                    <div class="form-label-custom mb-1">
                                        <span>Petal Width (Rộng cánh)</span>
                                        <span class="text-info fw-800" id="val_pw">0.2 cm</span>
                                    </div>
                                    <div class="input-group mb-1">
                                        <button class="btn btn-step" onclick="stepVal('petal_width', -0.1)">-</button>
                                        <input type="number" class="form-control custom-input" id="petal_width" value="0.2" step="0.1" oninput="syncSlider('petal_width', 'range_pw', 'val_pw')">
                                        <button class="btn btn-step" onclick="stepVal('petal_width', 0.1)">+</button>
                                    </div>
                                    <input type="range" class="form-range" id="range_pw" min="0.1" max="2.5" step="0.1" value="0.2" oninput="syncInput('petal_width', 'range_pw', 'val_pw')">
                                </div>

                                <button class="btn btn-analyze w-100 py-3 mt-2" onclick="makePrediction()">
                                    ⚡ CHẠY PHÂN TÍCH AI NOW
                                </button>
                            </div>
                        </div>

                        <!-- Result Display -->
                        <div class="col-xl-8 col-lg-7">
                            <div class="glass-card p-4 h-100">
                                <div id="placeholderText" class="text-center py-5 my-auto">
                                    <div class="display-1 mb-3">📊</div>
                                    <h3 class="fw-700">Đang chờ thông số đầu vào...</h3>
                                    <p class="text-sub fs-5">Bấm nút "CHẠY PHÂN TÍCH AI NOW" để xem kết quả.</p>
                                </div>

                                <div id="resultSection" style="display: none;">
                                    <div class="d-flex justify-content-between align-items-center mb-3">
                                        <div>
                                            <span id="flowerBadge" class="badge bg-warning text-dark fs-6 mb-1"></span>
                                            <h1 id="flowerName" class="flower-title text-uppercase m-0"></h1>
                                        </div>
                                    </div>

                                    <div class="row g-3">
                                        <div class="col-md-6">
                                            <div class="img-box-large mb-3">
                                                <img id="flowerImg" src="" alt="Specimen Image">
                                            </div>
                                            <div class="stat-box mb-2">
                                                <small class="text-sub d-block fw-600">Đặc điểm hình thái:</small>
                                                <span id="flowerDesc" class="fw-600 text-main fs-6"></span>
                                            </div>
                                        </div>

                                        <div class="col-md-6 d-flex flex-column justify-content-between">
                                            <div>
                                                <h6 class="fw-800 text-warning mb-2">Xác Suất Nhận Diện (Confidence)</h6>
                                                <div class="mb-2">
                                                    <div class="d-flex justify-content-between fw-700 mb-1 small">
                                                        <span>Iris Setosa</span>
                                                        <span id="probSetosa" class="text-info">0%</span>
                                                    </div>
                                                    <div class="progress progress-custom">
                                                        <div id="barSetosa" class="progress-bar bg-info" style="width: 0%"></div>
                                                    </div>
                                                </div>

                                                <div class="mb-2">
                                                    <div class="d-flex justify-content-between fw-700 mb-1 small">
                                                        <span>Iris Versicolor</span>
                                                        <span id="probVersicolor" class="text-warning">0%</span>
                                                    </div>
                                                    <div class="progress progress-custom">
                                                        <div id="barVersicolor" class="progress-bar bg-warning" style="width: 0%"></div>
                                                    </div>
                                                </div>

                                                <div class="mb-3">
                                                    <div class="d-flex justify-content-between fw-700 mb-1 small">
                                                        <span>Iris Virginica</span>
                                                        <span id="probVirginica" style="color:#ff007f;">0%</span>
                                                    </div>
                                                    <div class="progress progress-custom">
                                                        <div id="barVirginica" class="progress-bar" style="width: 0%; background-color: #ff007f;"></div>
                                                    </div>
                                                </div>
                                            </div>

                                            <div>
                                                <h6 class="fw-800 text-warning mb-1">Biểu Đồ Radar Khung Hình Học</h6>
                                                <div style="height: 180px; position: relative;">
                                                    <canvas id="radarChart"></canvas>
                                                </div>
                                            </div>
                                        </div>

                                        <div class="col-12 mt-2">
                                            <div class="row g-2">
                                                <div class="col-md-4">
                                                    <div class="stat-box text-center">
                                                        <small class="text-sub d-block">Tỷ lệ Cánh hoa (L/W)</small>
                                                        <span id="metricPetalRatio" class="fs-5 fw-800 text-info">--</span>
                                                    </div>
                                                </div>
                                                <div class="col-md-4">
                                                    <div class="stat-box text-center">
                                                        <small class="text-sub d-block">Diện tích Cánh ước tính</small>
                                                        <span id="metricPetalArea" class="fs-5 fw-800 text-warning">--</span>
                                                    </div>
                                                </div>
                                                <div class="col-md-4">
                                                    <div class="stat-box text-center">
                                                        <small class="text-sub d-block">Môi trường sống</small>
                                                        <span id="habitatVal" class="fs-6 fw-700 text-main">--</span>
                                                    </div>
                                                </div>
                                            </div>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>

                <!-- TAB 2: LIBRARY -->
                <div class="tab-pane fade" id="tab-library">
                    <div class="row g-4 py-2">
                        <div class="col-md-4">
                            <div class="glass-card p-4 text-center h-100">
                                <img src="https://upload.wikimedia.org/wikipedia/commons/5/56/Kosaciec_szczecinkowaty_Iris_setosa.jpg" class="img-fluid rounded-4 mb-3" style="height: 200px; object-fit: cover; width: 100%;">
                                <h4 class="fw-800 text-info">Iris Setosa</h4>
                                <p class="text-sub fs-6">Kích thước cánh hoa nhỏ nhất. Phân tách hoàn toàn tuyến tính với 2 loài còn lại.</p>
                            </div>
                        </div>
                        <div class="col-md-4">
                            <div class="glass-card p-4 text-center h-100">
                                <img src="https://upload.wikimedia.org/wikipedia/commons/4/41/Iris_versicolor_3.jpg" class="img-fluid rounded-4 mb-3" style="height: 200px; object-fit: cover; width: 100%;">
                                <h4 class="fw-800 text-warning">Iris Versicolor</h4>
                                <p class="text-sub fs-6">Loài có kích thước trung bình. Thường nằm ở vùng ranh giới quyết định (decision boundary).</p>
                            </div>
                        </div>
                        <div class="col-md-4">
                            <div class="glass-card p-4 text-center h-100">
                                <img src="https://upload.wikimedia.org/wikipedia/commons/9/9f/Iris_virginica.jpg" class="img-fluid rounded-4 mb-3" style="height: 200px; object-fit: cover; width: 100%;">
                                <h4 class="fw-800" style="color: #ff007f;">Iris Virginica</h4>
                                <p class="text-sub fs-6">Có kích thước cánh hoa và đài hoa lớn nhất trong ba loài Iris chuẩn.</p>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>

        <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
        <script>
            let radarChart = null;

            function toggleTheme() {
                const current = document.documentElement.getAttribute('data-bs-theme');
                const target = current === 'dark' ? 'light' : 'dark';
                document.documentElement.setAttribute('data-bs-theme', target);
            }

            function syncSlider(inputId, sliderId, labelId) {
                const val = parseFloat(document.getElementById(inputId).value) || 0;
                document.getElementById(sliderId).value = val;
                document.getElementById(labelId).innerText = val.toFixed(1) + " cm";
            }

            function syncInput(inputId, sliderId, labelId) {
                const val = parseFloat(document.getElementById(sliderId).value) || 0;
                document.getElementById(inputId).value = val;
                document.getElementById(labelId).innerText = val.toFixed(1) + " cm";
            }

            function stepVal(inputId, step) {
                const input = document.getElementById(inputId);
                let current = parseFloat(input.value) || 0;
                current = Math.max(0, parseFloat((current + step).toFixed(1)));
                input.value = current;
                
                if (inputId === 'sepal_length') syncSlider('sepal_length', 'range_sl', 'val_sl');
                if (inputId === 'sepal_width') syncSlider('sepal_width', 'range_sw', 'val_sw');
                if (inputId === 'petal_length') syncSlider('petal_length', 'range_pl', 'val_pl');
                if (inputId === 'petal_width') syncSlider('petal_width', 'range_pw', 'val_pw');
            }

            function loadPreset(sl, sw, pl, pw) {
                document.getElementById('sepal_length').value = sl;
                document.getElementById('sepal_width').value = sw;
                document.getElementById('petal_length').value = pl;
                document.getElementById('petal_width').value = pw;

                syncSlider('sepal_length', 'range_sl', 'val_sl');
                syncSlider('sepal_width', 'range_sw', 'val_sw');
                syncSlider('petal_length', 'range_pl', 'val_pl');
                syncSlider('petal_width', 'range_pw', 'val_pw');

                makePrediction();
            }

            async function makePrediction() {
                const sl = parseFloat(document.getElementById('sepal_length').value);
                const sw = parseFloat(document.getElementById('sepal_width').value);
                const pl = parseFloat(document.getElementById('petal_length').value);
                const pw = parseFloat(document.getElementById('petal_width').value);

                try {
                    const response = await fetch('/predict', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ sepal_length: sl, sepal_width: sw, petal_length: pl, petal_width: pw })
                    });
                    
                    const data = await response.json();
                    
                    document.getElementById('placeholderText').style.display = 'none';
                    document.getElementById('resultSection').style.display = 'block';

                    document.getElementById('flowerName').innerText = data.name;
                    document.getElementById('flowerBadge').innerText = data.badge;
                    document.getElementById('flowerImg').src = data.img;
                    document.getElementById('flowerDesc').innerText = data.desc;
                    document.getElementById('habitatVal').innerText = data.habitat;

                    const probs = data.calculated_probs;
                    document.getElementById('probSetosa').innerText = probs[0] + "%";
                    document.getElementById('barSetosa').style.width = probs[0] + "%";
                    
                    document.getElementById('probVersicolor').innerText = probs[1] + "%";
                    document.getElementById('barVersicolor').style.width = probs[1] + "%";
                    
                    document.getElementById('probVirginica').innerText = probs[2] + "%";
                    document.getElementById('barVirginica').style.width = probs[2] + "%";

                    // Metrics calculation
                    const ratio = pw > 0 ? (pl / pw).toFixed(2) : "--";
                    const area = (pl * pw * Math.PI / 4).toFixed(2);
                    document.getElementById('metricPetalRatio').innerText = ratio;
                    document.getElementById('metricPetalArea').innerText = area + " cm²";

                    updateChart(sl, sw, pl, pw);
                } catch (err) {
                    console.error("Lỗi khi kết nối API dự đoán:", err);
                }
            }

            function updateChart(sl, sw, pl, pw) {
                const ctx = document.getElementById('radarChart').getContext('2d');
                if (radarChart) radarChart.destroy();

                radarChart = new Chart(ctx, {
                    type: 'radar',
                    data: {
                        labels: ['Sepal Length', 'Sepal Width', 'Petal Length', 'Petal Width'],
                        datasets: [{
                            label: 'Kích thước (cm)',
                            data: [sl, sw, pl, pw],
                            backgroundColor: 'rgba(0, 242, 254, 0.25)',
                            borderColor: '#00f2fe',
                            borderWidth: 2,
                            pointBackgroundColor: '#ff007f'
                        }]
                    },
                    options: {
                        responsive: true,
                        maintainAspectRatio: false,
                        scales: {
                            r: {
                                min: 0,
                                max: 8,
                                ticks: { display: false },
                                grid: { color: 'rgba(255, 255, 255, 0.15)' }
                            }
                        },
                        plugins: { legend: { display: false } }
                    }
                });
            }
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)
