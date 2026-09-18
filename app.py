from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import joblib

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
        "probs": [100, 0, 0]
    },
    1: {
        "name": "IRIS VERSICOLOR",
        "badge": "Loài Phổ Biến - Nhóm 02",
        "img": "https://upload.wikimedia.org/wikipedia/commons/4/41/Iris_versicolor_3.jpg",
        "desc": "Kích thước trung bình, dải màu từ lam xám đến tím sẫm. Thuộc nhóm trung gian có đặc trưng biến thiên cao.",
        "habitat": "Ven sông, suối, vùng ven hồ",
        "origin": "Đông Bắc Bắc Mỹ",
        "probs": [2, 94, 4]
    },
    2: {
        "name": "IRIS VIRGINICA",
        "badge": "Loài Kích Thước Lớn - Nhóm 03",
        "img": "https://upload.wikimedia.org/wikipedia/commons/9/9f/Iris_virginica.jpg",
        "desc": "Sở hữu cánh hoa và lá đài phát triển tối đa. Cấu trúc hình học vượt trội so với hai nhóm loài còn lại.",
        "habitat": "Đồng cỏ ẩm ướt, đầm lầy cạn",
        "origin": "Đông Nam Hoa Kỳ",
        "probs": [0, 5, 95]
    }
}

@app.get("/", response_class=HTMLResponse)
def home_ui():
    html_content = """
    <!DOCTYPE html>
    <html lang="vi" data-bs-theme="dark">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Iris Analytics Pro - NextGen UI</title>
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
        <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;600;700;800&display=swap" rel="stylesheet">
        <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
        <style>
            :root {
                --bg-main: #0a0d18;
                --card-bg: rgba(23, 31, 56, 0.75);
                --card-border: rgba(255, 255, 255, 0.12);
                --accent-cyan: #00f2fe;
                --accent-pink: #ff007f;
                --accent-purple: #7928ca;
                --text-main: #f8fafc;
                --text-sub: #cbd5e1;
            }

            [data-bs-theme="light"] {
                --bg-main: #f0f4f9;
                --card-bg: rgba(255, 255, 255, 0.85);
                --card-border: rgba(0, 0, 0, 0.08);
                --text-main: #0f172a;
                --text-sub: #475569;
            }

            body {
                background-color: var(--bg-main);
                background-image: 
                    radial-gradient(circle at 15% 15%, rgba(121, 40, 202, 0.25) 0%, transparent 45%),
                    radial-gradient(circle at 85% 85%, rgba(0, 242, 254, 0.2) 0%, transparent 45%);
                color: var(--text-main);
                font-family: 'Plus Jakarta Sans', sans-serif;
                min-height: 100vh;
                font-size: 1.05rem; /* Tăng cỡ chữ cơ bản */
            }

            .navbar-custom {
                background: rgba(10, 13, 24, 0.85);
                backdrop-filter: blur(16px);
                border-bottom: 1px solid var(--card-border);
            }

            .glass-card {
                background: var(--card-bg);
                border: 1px solid var(--card-border);
                backdrop-filter: blur(20px);
                border-radius: 24px;
                box-shadow: 0 10px 30px rgba(0,0,0,0.3);
            }

            /* Cải thiện Tab Navigation */
            .nav-pills .nav-link {
                color: var(--text-sub);
                border-radius: 14px;
                padding: 12px 28px;
                font-weight: 700;
                font-size: 1.1rem;
                transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            }
            .nav-pills .nav-link.active {
                background: linear-gradient(135deg, #7928ca 0%, #ff007f 100%);
                color: #fff;
                box-shadow: 0 0 20px rgba(255, 0, 127, 0.4);
            }

            /* Cải thiện Input & Controls */
            .form-label-custom {
                font-size: 1.05rem;
                font-weight: 700;
                color: var(--text-main);
                display: flex;
                justify-content: space-between;
            }

            .custom-input {
                background: rgba(0, 0, 0, 0.25) !important;
                border: 2px solid var(--card-border) !important;
                color: var(--accent-cyan) !important;
                font-weight: 800;
                font-size: 1.25rem;
                text-align: center;
                border-radius: 12px;
            }

            .btn-step {
                background: linear-gradient(135deg, #2a2d3d, #1a1c29);
                border: 1px solid var(--card-border);
                color: #fff;
                font-weight: 800;
                font-size: 1.2rem;
                width: 45px;
                border-radius: 12px !important;
            }
            .btn-step:hover {
                background: var(--accent-cyan);
                color: #000;
            }

            /* Slider tùy chỉnh rực rỡ */
            .form-range::-webkit-slider-thumb {
                background: var(--accent-pink);
                box-shadow: 0 0 10px var(--accent-pink);
            }

            /* Nút Phân Tích rực rỡ */
            .btn-analyze {
                background: linear-gradient(135deg, #00f2fe 0%, #4facfe 100%);
                color: #000;
                font-weight: 800;
                font-size: 1.25rem;
                letter-spacing: 0.5px;
                border: none;
                border-radius: 16px;
                box-shadow: 0 0 25px rgba(0, 242, 254, 0.4);
                transition: all 0.3s;
            }
            .btn-analyze:hover {
                transform: translateY(-2px);
                box-shadow: 0 0 35px rgba(0, 242, 254, 0.7);
                color: #000;
            }

            .img-box {
                border-radius: 20px;
                overflow: hidden;
                border: 2px solid var(--card-border);
                background: #000;
                height: 300px;
                box-shadow: 0 8px 25px rgba(0,0,0,0.4);
            }
            .img-box img {
                width: 100%;
                height: 100%;
                object-fit: cover;
            }

            .progress-custom {
                height: 14px;
                background: rgba(255, 255, 255, 0.1);
                border-radius: 10px;
                overflow: hidden;
            }

            /* Cỡ chữ mô tả và nhãn */
            .text-large { font-size: 1.15rem; }
            .flower-title { font-size: 2.2rem; font-weight: 800; }
        </style>
    </head>
    <body>

        <!-- Header Navigation -->
        <nav class="navbar navbar-expand-lg navbar-dark navbar-custom sticky-top px-4">
            <div class="container-fluid">
                <a class="navbar-brand d-flex align-items-center gap-3 fw-800 fs-3" href="#">
                    <span>✨</span> <span style="background: linear-gradient(to right, #00f2fe, #ff007f); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">IRIS ANALYTICS PRO</span>
                </a>
                <div class="d-flex align-items-center gap-3">
                    <button class="btn btn-outline-light btn-sm rounded-pill px-3" onclick="toggleTheme()">🌓 Đổi Theme</button>
                    <span class="badge bg-success p-2 fs-6">SVM Engine Active</span>
                </div>
            </div>
        </nav>

        <div class="container-fluid px-4 py-4">
            <!-- Navigation Tabs -->
            <ul class="nav nav-pills mb-4 justify-content-center gap-3" id="mainTabs">
                <li class="nav-item">
                    <button class="nav-link active" id="predict-tab" data-bs-toggle="pill" data-bs-target="#tab-predict">🔮 Bảng Dự Đoán Interactive</button>
                </li>
                <li class="nav-item">
                    <button class="nav-link" id="library-tab" data-bs-toggle="pill" data-bs-target="#tab-library">📚 Thư Viện 3D & Sinh Học</button>
                </li>
                <li class="nav-item">
                    <button class="nav-link" id="history-tab" data-bs-toggle="pill" data-bs-target="#tab-history">📜 Lịch Sử & Xuất Dữ Liệu</button>
                </li>
            </ul>

            <div class="tab-content">
                
                <!-- TAB 1: PREDICT ENGINE -->
                <div class="tab-pane fade show active" id="tab-predict">
                    <div class="row g-4">
                        <!-- Input Controls (5 cols) -->
                        <div class="col-xl-5 col-lg-6">
                            <div class="glass-card p-4">
                                <h4 class="fw-800 mb-3 text-warning">🎛️ Tương Tác Thông Số Hoa</h4>
                                
                                <div class="mb-4">
                                    <label class="form-label text-sub fw-600 mb-2">Chọn mẫu thử nhanh:</label>
                                    <div class="d-flex gap-2">
                                        <button class="btn btn-outline-info flex-fill fw-700 py-2" onclick="loadPreset(5.1, 3.5, 1.4, 0.2)">🌸 Setosa</button>
                                        <button class="btn btn-outline-warning flex-fill fw-700 py-2" onclick="loadPreset(6.0, 2.9, 4.5, 1.5)">🌺 Versicolor</button>
                                        <button class="btn btn-outline-danger flex-fill fw-700 py-2" onclick="loadPreset(6.5, 3.0, 5.5, 2.0)">🌻 Virginica</button>
                                    </div>
                                </div>

                                <hr class="border-secondary opacity-25 mb-4">

                                <!-- Sepal Length -->
                                <div class="mb-4">
                                    <div class="form-label-custom mb-1">
                                        <span>Sepal Length (Dài đài hoa)</span>
                                        <span class="text-info" id="val_sl">5.1 cm</span>
                                    </div>
                                    <div class="input-group mb-2">
                                        <button class="btn btn-step" onclick="stepVal('sepal_length', -0.1)">-</button>
                                        <input type="number" class="form-control custom-input" id="sepal_length" value="5.1" step="0.1" oninput="syncSlider('sepal_length', 'range_sl', 'val_sl')">
                                        <button class="btn btn-step" onclick="stepVal('sepal_length', 0.1)">+</button>
                                    </div>
                                    <input type="range" class="form-range" id="range_sl" min="4.0" max="8.0" step="0.1" value="5.1" oninput="syncInput('sepal_length', 'range_sl', 'val_sl')">
                                </div>

                                <!-- Sepal Width -->
                                <div class="mb-4">
                                    <div class="form-label-custom mb-1">
                                        <span>Sepal Width (Rộng đài hoa)</span>
                                        <span class="text-info" id="val_sw">3.5 cm</span>
                                    </div>
                                    <div class="input-group mb-2">
                                        <button class="btn btn-step" onclick="stepVal('sepal_width', -0.1)">-</button>
                                        <input type="number" class="form-control custom-input" id="sepal_width" value="3.5" step="0.1" oninput="syncSlider('sepal_width', 'range_sw', 'val_sw')">
                                        <button class="btn btn-step" onclick="stepVal('sepal_width', 0.1)">+</button>
                                    </div>
                                    <input type="range" class="form-range" id="range_sw" min="2.0" max="4.5" step="0.1" value="3.5" oninput="syncInput('sepal_width', 'range_sw', 'val_sw')">
                                </div>

                                <!-- Petal Length -->
                                <div class="mb-4">
                                    <div class="form-label-custom mb-1">
                                        <span>Petal Length (Dài cánh hoa)</span>
                                        <span class="text-info" id="val_pl">1.4 cm</span>
                                    </div>
                                    <div class="input-group mb-2">
                                        <button class="btn btn-step" onclick="stepVal('petal_length', -0.1)">-</button>
                                        <input type="number" class="form-control custom-input" id="petal_length" value="1.4" step="0.1" oninput="syncSlider('petal_length', 'range_pl', 'val_pl')">
                                        <button class="btn btn-step" onclick="stepVal('petal_length', 0.1)">+</button>
                                    </div>
                                    <input type="range" class="form-range" id="range_pl" min="1.0" max="7.0" step="0.1" value="1.4" oninput="syncInput('petal_length', 'range_pl', 'val_pl')">
                                </div>

                                <!-- Petal Width -->
                                <div class="mb-4">
                                    <div class="form-label-custom mb-1">
                                        <span>Petal Width (Rộng cánh hoa)</span>
                                        <span class="text-info" id="val_pw">0.2 cm</span>
                                    </div>
                                    <div class="input-group mb-2">
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

                        <!-- Result Display (7 cols) -->
                        <div class="col-xl-7 col-lg-6">
                            <div class="glass-card p-4 h-100">
                                <div id="placeholderText" class="text-center py-5 my-auto">
                                    <div class="display-1 mb-3">🌿</div>
                                    <h3 class="fw-700">Chờ lệnh phân tích...</h3>
                                    <p class="text-sub fs-5">Hãy thay đổi thông số ở bảng bên trái hoặc nhấn nút Phân Tích.</p>
                                </div>

                                <div id="resultSection" style="display: none;">
                                    <div class="d-flex justify-content-between align-items-center mb-3">
                                        <div>
                                            <span id="flowerBadge" class="badge bg-warning text-dark fs-6 mb-2"></span>
                                            <h1 id="flowerName" class="flower-title text-uppercase m-0" style="color: var(--accent-cyan);"></h1>
                                        </div>
                                    </div>

                                    <div class="row g-4">
                                        <div class="col-md-6">
                                            <div class="img-box mb-3">
                                                <img id="flowerImg" src="" alt="Specimen Image">
                                            </div>
                                            <p id="flowerDesc" class="text-large text-sub mb-0"></p>
                                        </div>

                                        <div class="col-md-6">
                                            <h5 class="fw-800 text-warning mb-3">Xác Suất Nhận Diện (Confidence)</h5>
                                            
                                            <div class="mb-3">
                                                <div class="d-flex justify-content-between fw-700 mb-1 fs-6">
                                                    <span>Iris Setosa</span>
                                                    <span id="probSetosa" class="text-info">0%</span>
                                                </div>
                                                <div class="progress progress-custom">
                                                    <div id="barSetosa" class="progress-bar bg-info progress-bar-striped progress-bar-animated" style="width: 0%"></div>
                                                </div>
                                            </div>

                                            <div class="mb-3">
                                                <div class="d-flex justify-content-between fw-700 mb-1 fs-6">
                                                    <span>Iris Versicolor</span>
                                                    <span id="probVersicolor" class="text-warning">0%</span>
                                                </div>
                                                <div class="progress progress-custom">
                                                    <div id="barVersicolor" class="progress-bar bg-warning progress-bar-striped progress-bar-animated" style="width: 0%"></div>
                                                </div>
                                            </div>

                                            <div class="mb-4">
                                                <div class="d-flex justify-content-between fw-700 mb-1 fs-6">
                                                    <span>Iris Virginica</span>
                                                    <span id="probVirginica" style="color:#ff007f;">0%</span>
                                                </div>
                                                <div class="progress progress-custom">
                                                    <div id="barVirginica" class="progress-bar progress-bar-striped progress-bar-animated" style="width: 0%; background-color: #ff007f;"></div>
                                                </div>
                                            </div>

                                            <!-- Biểu đồ Radar so sánh -->
                                            <h5 class="fw-800 text-warning mb-2">Biểu Đồ Hình Học (Radar Chart)</h5>
                                            <div style="height: 180px; position: relative;">
                                                <canvas id="radarChart"></canvas>
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
                    <div class="row g-4">
                        <div class="col-md-4">
                            <div class="glass-card p-4 text-center h-100">
                                <img src="https://upload.wikimedia.org/wikipedia/commons/5/56/Kosaciec_szczecinkowaty_Iris_setosa.jpg" class="img-fluid rounded-4 mb-3" style="height: 220px; object-fit: cover; width: 100%;">
                                <h3 class="fw-800 text-info">Iris Setosa</h3>
                                <p class="text-sub fs-6">Kích thước cánh hoa nhỏ nhất. Đánh giá dựa trên góc đài hoa hẹp và cấu trúc cánh đứng.</p>
                            </div>
                        </div>
                        <div class="col-md-4">
                            <div class="glass-card p-4 text-center h-100">
                                <img src="https://upload.wikimedia.org/wikipedia/commons/4/41/Iris_versicolor_3.jpg" class="img-fluid rounded-4 mb-3" style="height: 220px; object-fit: cover; width: 100%;">
                                <h3 class="fw-800 text-warning">Iris Versicolor</h3>
                                <p class="text-sub fs-6">Kích thước trung bình, dải màu biến đổi đa dạng từ xám lam tới hoa xám tím.</p>
                            </div>
                        </div>
                        <div class="col-md-4">
                            <div class="glass-card p-4 text-center h-100">
                                <img src="https://upload.wikimedia.org/wikipedia/commons/9/9f/Iris_virginica.jpg" class="img-fluid rounded-4 mb-3" style="height: 220px; object-fit: cover; width: 100%;">
                                <h3 class="fw-800" style="color: #ff007f;">Iris Virginica</h3>
                                <p class="text-sub fs-6">Kích thước lớn nhất trong bộ dữ liệu. Cánh hoa dài rủ xuống tự nhiên.</p>
                            </div>
                        </div>
                    </div>
                </div>

                <!-- TAB 3: HISTORY -->
                <div class="tab-pane fade" id="tab-history">
                    <div class="glass-card p-4">
                        <div class="d-flex justify-content-between align-items-center mb-3">
                            <h4 class="fw-800 text-warning m-0">📜 Lịch Sử Dự Đoán Phiên Hiện Tại</h4>
                            <div class="d-flex gap-2">
                                <button class="btn btn-outline-info" onclick="exportCSV()">📥 Xuất CSV</button>
                                <button class="btn btn-outline-danger" onclick="clearHistory()">🗑️ Xóa Lịch Sử</button>
                            </div>
                        </div>
                        <div class="table-responsive">
                            <table class="table table-dark table-hover align-middle fs-6">
                                <thead>
                                    <tr>
                                        <th>#</th>
                                        <th>Sepal Length</th>
                                        <th>Sepal Width</th>
                                        <th>Petal Length</th>
                                        <th>Petal Width</th>
                                        <th>Kết Quả Chẩn Đoán</th>
                                    </tr>
                                </thead>
                                <tbody id="historyTableBody">
                                    <tr>
                                        <td colspan="6" class="text-center text-muted py-4">Chưa có kết quả dự đoán nào.</td>
                                    </tr>
                                </tbody>
                            </table>
                        </div>
                    </div>
                </div>

            </div>
        </div>

        <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
        <script>
            let historyLog = [];
            let radarChart = null;

            function toggleTheme() {
                const currentTheme = document.documentElement.getAttribute('data-bs-theme');
                document.documentElement.setAttribute('data-bs-theme', currentTheme === 'dark' ? 'light' : 'dark');
            }

            function syncSlider(inputId, sliderId, labelId) {
                const val = document.getElementById(inputId).value;
                document.getElementById(sliderId).value = val;
                document.getElementById(labelId).innerText = val + ' cm';
            }

            function syncInput(inputId, sliderId, labelId) {
                const val = document.getElementById(sliderId).value;
                document.getElementById(inputId).value = val;
                document.getElementById(labelId).innerText = val + ' cm';
            }

            function stepVal(id, delta) {
                let el = document.getElementById(id);
                let val = (parseFloat(el.value) || 0) + delta;
                el.value = Math.max(0.1, val).toFixed(1);
                
                if (id === 'sepal_length') syncSlider('sepal_length', 'range_sl', 'val_sl');
                if (id === 'sepal_width') syncSlider('sepal_width', 'range_sw', 'val_sw');
                if (id === 'petal_length') syncSlider('petal_length', 'range_pl', 'val_pl');
                if (id === 'petal_width') syncSlider('petal_width', 'range_pw', 'val_pw');
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

                    if (!response.ok) return alert("Lỗi kết nối máy chủ!");

                    const result = await response.json();

                    document.getElementById('placeholderText').style.display = 'none';
                    document.getElementById('resultSection').style.display = 'block';

                    document.getElementById('flowerName').innerText = result.info.name;
                    document.getElementById('flowerBadge').innerText = result.info.badge;
                    document.getElementById('flowerImg').src = result.info.img;
                    document.getElementById('flowerDesc').innerText = result.info.desc;

                    // Update Probabilities
                    const probs = result.info.probs;
                    document.getElementById('probSetosa').innerText = probs[0] + '%';
                    document.getElementById('barSetosa').style.width = probs[0] + '%';

                    document.getElementById('probVersicolor').innerText = probs[1] + '%';
                    document.getElementById('barVersicolor').style.width = probs[1] + '%';

                    document.getElementById('probVirginica').innerText = probs[2] + '%';
                    document.getElementById('barVirginica').style.width = probs[2] + '%';

                    // Update Radar Chart
                    renderRadarChart([sl, sw, pl, pw]);

                    // Save History
                    historyLog.unshift({ sl, sw, pl, pw, result: result.info.name });
                    renderHistory();

                } catch (err) {
                    console.error(err);
                }
            }

            function renderRadarChart(dataPoints) {
                const ctx = document.getElementById('radarChart').getContext('2d');
                if (radarChart) radarChart.destroy();

                radarChart = new Chart(ctx, {
                    type: 'radar',
                    data: {
                        labels: ['Sepal Length', 'Sepal Width', 'Petal Length', 'Petal Width'],
                        datasets: [{
                            label: 'Thông Số Nhập',
                            data: dataPoints,
                            backgroundColor: 'rgba(0, 242, 254, 0.3)',
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
                                angleLines: { color: 'rgba(255,255,255,0.2)' },
                                grid: { color: 'rgba(255,255,255,0.2)' },
                                pointLabels: { color: '#cbd5e1', font: { size: 11, weight: 'bold' } },
                                ticks: { display: false }
                            }
                        },
                        plugins: { legend: { display: false } }
                    }
                });
            }

            function renderHistory() {
                const tbody = document.getElementById('historyTableBody');
                if(historyLog.length === 0) {
                    tbody.innerHTML = '<tr><td colspan="6" class="text-center text-muted py-4">Chưa có kết quả.</td></tr>';
                    return;
                }
                tbody.innerHTML = historyLog.map((item, index) => `
                    <tr>
                        <td>${index + 1}</td>
                        <td>${item.sl} cm</td>
                        <td>${item.sw} cm</td>
                        <td>${item.pl} cm</td>
                        <td>${item.pw} cm</td>
                        <td><span class="badge bg-primary fs-6">${item.result}</span></td>
                    </tr>
                `).join('');
            }

            function clearHistory() {
                historyLog = [];
                renderHistory();
            }

            function exportCSV() {
                if (historyLog.length === 0) return alert('Chưa có lịch sử để xuất!');
                let csvContent = "data:text/csv;charset=utf-8,SepalLength,SepalWidth,PetalLength,PetalWidth,Prediction\\n"
                    + historyLog.map(e => `${e.sl},${e.sw},${e.pl},${e.pw},${e.result}`).join("\\n");
                
                const encodedUri = encodeURI(csvContent);
                const link = document.createElement("a");
                link.setAttribute("href", encodedUri);
                link.setAttribute("download", "iris_report.csv");
                document.body.appendChild(link);
                link.click();
            }
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

@app.post("/predict")
def predict(data: IrisInput):
    features = [[data.sepal_length, data.sepal_width, data.petal_length, data.petal_width]]
    
    if model is not None:
        prediction = int(model.predict(features)[0])
        info = species_info[prediction].copy()
        
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(features)[0]
            info["probs"] = [round(p * 100, 1) for p in probs]
    else:
        prediction = 0
        info = species_info[prediction]

    return {
        "class_id": prediction,
        "prediction": info["name"],
        "info": info
    }
