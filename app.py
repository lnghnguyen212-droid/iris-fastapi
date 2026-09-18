from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import joblib

# Nạp mô hình SVM đã huấn luyện
model = joblib.load("svm_model.pkl")

app = FastAPI(title="Iris AI Enterprise Suite")

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
        "desc": "Đặc trưng bởi lá đài rộng, cánh hoa nhỏ gọn. Mô hình SVM nhận diện loài này với độ tin cậy tuyệt đối 100%.",
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
    <html lang="vi">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Iris Analytics Pro Suite</title>
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
        <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;600;700;800&display=swap" rel="stylesheet">
        <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
        <style>
            :root {
                --bg-main: #0b0f19;
                --card-bg: rgba(22, 30, 49, 0.7);
                --card-border: rgba(255, 255, 255, 0.08);
                --accent-cyan: #38bdf8;
                --accent-purple: #c084fc;
            }

            body {
                background-color: var(--bg-main);
                background-image: 
                    radial-gradient(at 10% 10%, rgba(56, 189, 248, 0.08) 0px, transparent 50%),
                    radial-gradient(at 90% 90%, rgba(192, 132, 252, 0.08) 0px, transparent 50%);
                color: #f8fafc;
                font-family: 'Plus Jakarta Sans', sans-serif;
                min-height: 100vh;
            }

            .navbar-custom {
                background: rgba(11, 15, 25, 0.8);
                backdrop-filter: blur(12px);
                border-bottom: 1px solid var(--card-border);
            }

            .main-container {
                width: 100%;
                padding: 1.5rem 2rem;
            }

            .glass-card {
                background: var(--card-bg);
                border: 1px solid var(--card-border);
                backdrop-filter: blur(16px);
                border-radius: 20px;
            }

            .nav-pills .nav-link {
                color: #94a3b8;
                border-radius: 12px;
                padding: 10px 20px;
                font-weight: 600;
                transition: all 0.3s;
            }
            .nav-pills .nav-link.active {
                background: linear-gradient(135deg, #6366f1 0%, #a855f7 100%);
                color: #fff;
                box-shadow: 0 4px 15px rgba(168, 85, 247, 0.3);
            }

            .custom-input {
                background: rgba(11, 15, 25, 0.6) !important;
                border: 1px solid var(--card-border) !important;
                color: var(--accent-cyan) !important;
                font-weight: 700;
                text-align: center;
            }

            .btn-step {
                background: rgba(255, 255, 255, 0.05);
                border: 1px solid var(--card-border);
                color: #fff;
            }

            .img-box {
                border-radius: 16px;
                overflow: hidden;
                border: 1px solid var(--card-border);
                background: #000;
                height: 320px;
                display: flex;
                align-items: center;
                justify-content: center;
            }

            .img-box img {
                width: 100%;
                height: 100%;
                object-fit: contain;
            }

            .progress-custom {
                height: 8px;
                background: rgba(255, 255, 255, 0.05);
                border-radius: 4px;
            }

            .table-custom {
                color: #cbd5e1;
            }
            .table-custom th {
                color: #94a3b8;
                border-bottom-color: var(--card-border);
            }
            .table-custom td {
                border-bottom-color: rgba(255, 255, 255, 0.03);
            }
        </style>
    </head>
    <body>

        <!-- Header Navigation -->
        <nav class="navbar navbar-expand-lg navbar-dark navbar-custom sticky-top px-4">
            <div class="container-fluid">
                <a class="navbar-brand d-flex align-items-center gap-2 fw-800" href="#">
                    <span class="fs-4">🧬</span> IRIS ANALYTICS PRO
                </a>
                <div class="d-flex align-items-center gap-3">
                    <span class="badge bg-success bg-opacity-25 text-success border border-success border-opacity-25">Engine: Active (SVM)</span>
                </div>
            </div>
        </nav>

        <div class="main-container">
            <!-- Navigation Tabs -->
            <ul class="nav nav-pills mb-4 justify-content-center gap-2" id="mainTabs" role="tablist">
                <li class="nav-item">
                    <button class="nav-link active" id="predict-tab" data-bs-toggle="pill" data-bs-target="#tab-predict">🔮 Phân Loại AI</button>
                </li>
                <li class="nav-item">
                    <button class="nav-link" id="library-tab" data-bs-toggle="pill" data-bs-target="#tab-library">📚 Thư Viện Loài Hoa</button>
                </li>
                <li class="nav-item">
                    <button class="nav-link" id="history-tab" data-bs-toggle="pill" data-bs-target="#tab-history">📜 Lịch Sử Dự Đoán</button>
                </li>
            </ul>

            <div class="tab-content" id="mainTabsContent">
                
                <!-- TAB 1: PREDICT ENGINE -->
                <div class="tab-pane fade show active" id="tab-predict">
                    <div class="row g-4">
                        <!-- Input Controls (4 cols) -->
                        <div class="col-lg-4">
                            <div class="glass-card p-4 h-100">
                                <h5 class="fw-700 mb-3 text-light">🎛️ Bảng Điều Khiển</h5>
                                
                                <div class="mb-3">
                                    <label class="small text-secondary mb-1">Mẫu nhanh:</label>
                                    <div class="d-flex gap-1">
                                        <button class="btn btn-sm btn-outline-info flex-fill" onclick="loadPreset(5.1, 3.5, 1.4, 0.2)">Setosa</button>
                                        <button class="btn btn-sm btn-outline-info flex-fill" onclick="loadPreset(6.0, 2.9, 4.5, 1.5)">Versicolor</button>
                                        <button class="btn btn-sm btn-outline-info flex-fill" onclick="loadPreset(6.5, 3.0, 5.5, 2.0)">Virginica</button>
                                    </div>
                                </div>

                                <hr class="border-secondary opacity-25">

                                <div class="mb-3">
                                    <label class="small text-secondary mb-1">Sepal Length (Dài đài hoa):</label>
                                    <div class="input-group">
                                        <button class="btn btn-step" onclick="stepVal('sepal_length', -0.1)">-</button>
                                        <input type="number" class="form-control custom-input" id="sepal_length" value="5.1" step="0.1">
                                        <button class="btn btn-step" onclick="stepVal('sepal_length', 0.1)">+</button>
                                    </div>
                                </div>

                                <div class="mb-3">
                                    <label class="small text-secondary mb-1">Sepal Width (Rộng đài hoa):</label>
                                    <div class="input-group">
                                        <button class="btn btn-step" onclick="stepVal('sepal_width', -0.1)">-</button>
                                        <input type="number" class="form-control custom-input" id="sepal_width" value="3.5" step="0.1">
                                        <button class="btn btn-step" onclick="stepVal('sepal_width', 0.1)">+</button>
                                    </div>
                                </div>

                                <div class="mb-3">
                                    <label class="small text-secondary mb-1">Petal Length (Dài cánh hoa):</label>
                                    <div class="input-group">
                                        <button class="btn btn-step" onclick="stepVal('petal_length', -0.1)">-</button>
                                        <input type="number" class="form-control custom-input" id="petal_length" value="1.4" step="0.1">
                                        <button class="btn btn-step" onclick="stepVal('petal_length', 0.1)">+</button>
                                    </div>
                                </div>

                                <div class="mb-4">
                                    <label class="small text-secondary mb-1">Petal Width (Rộng cánh hoa):</label>
                                    <div class="input-group">
                                        <button class="btn btn-step" onclick="stepVal('petal_width', -0.1)">-</button>
                                        <input type="number" class="form-control custom-input" id="petal_width" value="0.2" step="0.1">
                                        <button class="btn btn-step" onclick="stepVal('petal_width', 0.1)">+</button>
                                    </div>
                                </div>

                                <button class="btn btn-primary w-100 py-3 fw-700 text-uppercase rounded-3" style="background: linear-gradient(135deg, #6366f1, #a855f7); border:none;" onclick="makePrediction()">
                                    Phân Tích Dữ Liệu
                                </button>
                            </div>
                        </div>

                        <!-- Result Display (8 cols) -->
                        <div class="col-lg-8">
                            <div class="glass-card p-4 h-100">
                                <div id="placeholderText" class="text-center py-5 my-auto">
                                    <div class="display-3 mb-3 text-muted">📊</div>
                                    <h5>Đang chờ thông số đầu vào...</h5>
                                    <p class="text-secondary small">Nhấn "Phân Tích Dữ Liệu" để xem kết quả toàn diện.</p>
                                </div>

                                <div id="resultSection" style="display: none;">
                                    <div class="d-flex justify-content-between align-items-center mb-3">
                                        <div>
                                            <span id="flowerBadge" class="badge bg-info bg-opacity-25 text-info border border-info border-opacity-25 mb-1"></span>
                                            <h3 id="flowerName" class="fw-800 text-light m-0"></h3>
                                        </div>
                                        <button class="btn btn-sm btn-outline-light" onclick="exportCSV()">📥 Xuất Kết Quả CSV</button>
                                    </div>

                                    <div class="row g-3">
                                        <div class="col-md-6">
                                            <div class="img-box mb-3">
                                                <img id="flowerImg" src="" alt="Specimen Image">
                                            </div>
                                            <p id="flowerDesc" class="small text-secondary mb-0"></p>
                                        </div>

                                        <div class="col-md-6">
                                            <h6 class="fw-700 text-light mb-3">Phân Tích Xác Suất (Confidence)</h6>
                                            
                                            <div class="mb-3">
                                                <div class="d-flex justify-content-between small mb-1">
                                                    <span>Iris Setosa</span>
                                                    <span id="probSetosa">0%</span>
                                                </div>
                                                <div class="progress progress-custom">
                                                    <div id="barSetosa" class="progress-bar bg-info" style="width: 0%"></div>
                                                </div>
                                            </div>

                                            <div class="mb-3">
                                                <div class="d-flex justify-content-between small mb-1">
                                                    <span>Iris Versicolor</span>
                                                    <span id="probVersicolor">0%</span>
                                                </div>
                                                <div class="progress progress-custom">
                                                    <div id="barVersicolor" class="progress-bar bg-warning" style="width: 0%"></div>
                                                </div>
                                            </div>

                                            <div class="mb-4">
                                                <div class="d-flex justify-content-between small mb-1">
                                                    <span>Iris Virginica</span>
                                                    <span id="probVirginica">0%</span>
                                                </div>
                                                <div class="progress progress-custom">
                                                    <div id="barVirginica" class="progress-bar bg-purple" style="width: 0%; background-color: #c084fc;"></div>
                                                </div>
                                            </div>

                                            <h6 class="fw-700 text-light mb-2">Đặc Tính Sinh Học</h6>
                                            <div class="p-3 rounded-3 mb-2" style="background: rgba(0,0,0,0.2);">
                                                <small class="text-secondary d-block">Môi trường sống:</small>
                                                <span id="habitatVal" class="small text-light fw-600">--</span>
                                            </div>
                                            <div class="p-3 rounded-3" style="background: rgba(0,0,0,0.2);">
                                                <small class="text-secondary d-block">Phân bố tự nhiên:</small>
                                                <span id="originVal" class="small text-light fw-600">--</span>
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
                            <div class="glass-card p-3 h-100 text-center">
                                <img src="https://upload.wikimedia.org/wikipedia/commons/5/56/Kosaciec_szczecinkowaty_Iris_setosa.jpg" class="img-fluid rounded-3 mb-3" style="height: 200px; object-fit: cover; width: 100%;">
                                <h5 class="fw-700 text-info">Iris Setosa</h5>
                                <p class="small text-secondary">Kích thước cánh hoa nhỏ nhất, đài hoa rộng. Khả năng chống chịu thời tiết lạnh tốt nhất trong 3 loài.</p>
                            </div>
                        </div>
                        <div class="col-md-4">
                            <div class="glass-card p-3 h-100 text-center">
                                <img src="https://upload.wikimedia.org/wikipedia/commons/4/41/Iris_versicolor_3.jpg" class="img-fluid rounded-3 mb-3" style="height: 200px; object-fit: cover; width: 100%;">
                                <h5 class="fw-700 text-warning">Iris Versicolor</h5>
                                <p class="small text-secondary">Kích thước trung bình, dải sắc tố đa dạng từ xanh xám đến xanh tím. Thường phân bố ở vùng đầm lầy ven biển.</p>
                            </div>
                        </div>
                        <div class="col-md-4">
                            <div class="glass-card p-3 h-100 text-center">
                                <img src="https://upload.wikimedia.org/wikipedia/commons/9/9f/Iris_virginica.jpg" class="img-fluid rounded-3 mb-3" style="height: 200px; object-fit: cover; width: 100%;">
                                <h5 class="fw-700 text-purple" style="color: #c084fc;">Iris Virginica</h5>
                                <p class="small text-secondary">Loài có kích thước lớn nhất. Cánh hoa rủ xuống đặc trưng với sắc tím đậm ấn tượng.</p>
                            </div>
                        </div>
                    </div>
                </div>

                <!-- TAB 3: HISTORY -->
                <div class="tab-pane fade" id="tab-history">
                    <div class="glass-card p-4">
                        <div class="d-flex justify-content-between align-items-center mb-3">
                            <h5 class="fw-700 m-0">📜 Lịch Sử Phân Tích Trong Phiên</h5>
                            <button class="btn btn-sm btn-outline-danger" onclick="clearHistory()">Xóa Lịch Sử</button>
                        </div>
                        <div class="table-responsive">
                            <table class="table table-custom align-middle">
                                <thead>
                                    <tr>
                                        <th>#</th>
                                        <th>Sepal L</th>
                                        <th>Sepal W</th>
                                        <th>Petal L</th>
                                        <th>Petal W</th>
                                        <th>Kết Quả Dự Đoán</th>
                                    </tr>
                                </thead>
                                <tbody id="historyTableBody">
                                    <tr>
                                        <td colspan="6" class="text-center text-muted py-4">Chưa có dữ liệu phân tích nào được lưu.</td>
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

            function stepVal(id, delta) {
                let el = document.getElementById(id);
                let val = parseFloat(el.value) || 0;
                el.value = Math.max(0.1, (val + delta)).toFixed(1);
            }

            function loadPreset(sl, sw, pl, pw) {
                document.getElementById('sepal_length').value = sl;
                document.getElementById('sepal_width').value = sw;
                document.getElementById('petal_length').value = pl;
                document.getElementById('petal_width').value = pw;
                makePrediction();
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

                document.getElementById('placeholderText').style.display = 'none';
                document.getElementById('resultSection').style.display = 'block';

                document.getElementById('flowerName').innerText = result.info.name;
                document.getElementById('flowerBadge').innerText = result.info.badge;
                document.getElementById('flowerImg').src = result.info.img;
                document.getElementById('flowerDesc').innerText = result.info.desc;
                document.getElementById('habitatVal').innerText = result.info.habitat;
                document.getElementById('originVal').innerText = result.info.origin;

                // Probability Progress Bars
                const probs = result.info.probs;
                document.getElementById('probSetosa').innerText = probs[0] + '%';
                document.getElementById('barSetosa').style.width = probs[0] + '%';

                document.getElementById('probVersicolor').innerText = probs[1] + '%';
                document.getElementById('barVersicolor').style.width = probs[1] + '%';

                document.getElementById('probVirginica').innerText = probs[2] + '%';
                document.getElementById('barVirginica').style.width = probs[2] + '%';

                // Save to history
                historyLog.unshift({ sl, sw, pl, pw, result: result.info.name });
                renderHistory();
            }

            function renderHistory() {
                const tbody = document.getElementById('historyTableBody');
                if(historyLog.length === 0) {
                    tbody.innerHTML = '<tr><td colspan="6" class="text-center text-muted py-4">Chưa có dữ liệu.</td></tr>';
                    return;
                }
                tbody.innerHTML = historyLog.map((item, index) => `
                    <tr>
                        <td>${index + 1}</td>
                        <td>${item.sl} cm</td>
                        <td>${item.sw} cm</td>
                        <td>${item.pl} cm</td>
                        <td>${item.pw} cm</td>
                        <td><span class="badge bg-primary">${item.result}</span></td>
                    </tr>
                `).join('');
            }

            function clearHistory() {
                historyLog = [];
                renderHistory();
            }

            function exportCSV() {
                if (historyLog.length === 0) return alert('Chưa có lịch sử để xuất dữ liệu!');
                let csvContent = "data:text/csv;charset=utf-8,SepalLength,SepalWidth,PetalLength,PetalWidth,Prediction\\n"
                    + historyLog.map(e => `${e.sl},${e.sw},${e.pl},${e.pw},${e.result}`).join("\\n");
                
                const encodedUri = encodeURI(csvContent);
                const link = document.createElement("a");
                link.setAttribute("href", encodedUri);
                link.setAttribute("download", "iris_analysis_report.csv");
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
    prediction = int(model.predict(features)[0])
    
    return {
        "class_id": prediction,
        "prediction": species_info[prediction]["name"],
        "info": species_info[prediction]
    }
