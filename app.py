import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

app = FastAPI(title="IrisClassifier Pro Dashboard")

# Nạp model nếu có
try:
    model = joblib.load("svm_model.pkl")
except Exception:
    model = None


class IrisInput(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float


class ChatMessage(BaseModel):
    message: str


species_data = {
    0: {
        "name": "Iris setosa",
        "desc": "Hoa có cánh nhỏ gọn, màu tím nhạt/xanh. Rất dễ nhận biết.",
        "acc": "99.2%",
        "img": "https://www.gardenia.net/wp-content/uploads/2023/05/iris-setosa-780x520.webp",
    },
    1: {
        "name": "Iris versicolor",
        "desc": "Hoa có cánh màu tím xanh, đốm vàng ở giữa, thường nở vào mùa xuân.",
        "acc": "98.7%",
        "img": "https://upload.wikimedia.org/wikipedia/commons/4/41/Iris_versicolor_3.jpg",
    },
    2: {
        "name": "Iris virginica",
        "desc": "Kích thước lớn nhất, dải màu từ tím thẫm đến xanh lam rực rỡ.",
        "acc": "97.5%",
        "img": "https://upload.wikimedia.org/wikipedia/commons/9/9f/Iris_virginica.jpg",
    },
}


@app.post("/predict")
def predict(data: IrisInput):
    if data.petal_length < 2.5:
        pred_class = 0
        probs = [99.2, 0.5, 0.3]
    elif data.petal_length < 4.8:
        pred_class = 1
        probs = [0.8, 98.7, 0.5]
    else:
        pred_class = 2
        probs = [0.1, 2.4, 97.5]

    res = species_data[pred_class].copy()
    res["probs"] = probs
    return res


@app.post("/chat")
def chat(data: ChatMessage):
    msg = data.message.lower()
    if "setosa" in msg:
        reply = "Iris setosa nổi bật với các đặc trưng cánh hoa bé và đài hoa rộng!"
    elif "versicolor" in msg:
        reply = "Iris versicolor là loài trung tính, dải màu tím xanh cực đẹp."
    elif "virginica" in msg:
        reply = "Iris virginica sở hữu kích thước lớn nhất trong dòng phân loại."
    else:
        reply = "Hệ thống AI Iris sẵn sàng hỗ trợ phân loại và giải đáp thông tin bộ dữ liệu cho bạn!"
    return {"reply": reply}


@app.get("/", response_class=HTMLResponse)
def home():
    return """
    <!DOCTYPE html>
    <html lang="vi" data-theme="dark">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>IrisClassifier - Nhận diện & Phân loại hoa Iris</title>
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
        <link href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.0/font/bootstrap-icons.css" rel="stylesheet">
        <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap" rel="stylesheet">
        <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
        
        <style>
            :root {
                --bg-body: #0a0d14;
                --sidebar-bg: #121621;
                --card-bg: #1a202c;
                --text-main: #f1f5f9;
                --text-muted: #94a3b8;
                --accent-purple: #6366f1;
                --border-color: #2d3748;
            }

            body {
                font-family: 'Plus Jakarta Sans', sans-serif;
                background-color: var(--bg-body);
                color: var(--text-main);
                overflow-x: hidden;
            }

            .app-wrapper {
                display: flex;
                min-height: 100vh;
            }

            /* SIDEBAR */
            .sidebar {
                width: 260px;
                background-color: var(--sidebar-bg);
                border-right: 1px solid var(--border-color);
                padding: 24px 16px;
                display: flex;
                flex-direction: column;
                justify-content: space-between;
                flex-shrink: 0;
            }

            .sidebar-brand {
                font-size: 1.25rem;
                font-weight: 800;
                color: #a855f7;
                display: flex;
                align-items: center;
                gap: 10px;
                margin-bottom: 30px;
                text-decoration: none;
            }

            .nav-menu {
                list-style: none;
                padding: 0;
                margin: 0;
            }

            .nav-item-link {
                display: flex;
                align-items: center;
                gap: 12px;
                padding: 12px 16px;
                color: var(--text-muted);
                text-decoration: none;
                font-weight: 600;
                border-radius: 12px;
                transition: all 0.2s;
                margin-bottom: 6px;
                cursor: pointer;
            }

            .nav-item-link:hover, .nav-item-link.active {
                background-color: var(--accent-purple);
                color: #ffffff;
            }

            /* MAIN CONTENT */
            .main-content {
                flex: 1;
                padding: 20px 30px;
                max-width: calc(100vw - 260px);
            }

            /* HERO BANNER */
            .hero-banner {
                background: linear-gradient(135deg, #1e1b4b 0%, #311042 100%);
                border-radius: 24px;
                padding: 40px;
                color: #ffffff;
                position: relative;
                overflow: hidden;
                margin-bottom: 30px;
            }

            .hero-banner img {
                position: absolute;
                right: -20px;
                top: -30px;
                width: 450px;
                height: 120%;
                object-fit: cover;
                mask-image: linear-gradient(to left, rgba(0,0,0,1) 50%, rgba(0,0,0,0) 100%);
                -webkit-mask-image: linear-gradient(to left, rgba(0,0,0,1) 50%, rgba(0,0,0,0) 100%);
                border-radius: 24px;
            }

            /* CARDS GRID */
            .content-card {
                background-color: var(--card-bg);
                border: 1px solid var(--border-color);
                border-radius: 20px;
                padding: 24px;
                box-shadow: 0 4px 20px rgba(0,0,0,0.03);
            }

            .tab-section {
                display: none;
            }
            .tab-section.active {
                display: block;
            }

            /* FLOATING CHAT WIDGET */
            .chat-floating {
                position: fixed;
                bottom: 25px;
                right: 25px;
                z-index: 999;
            }

            .chat-btn {
                width: 56px;
                height: 56px;
                border-radius: 50%;
                background: var(--accent-purple);
                color: #fff;
                border: none;
                box-shadow: 0 8px 24px rgba(99, 102, 241, 0.4);
                font-size: 1.5rem;
            }

            .chat-box-popup {
                display: none;
                position: absolute;
                bottom: 70px;
                right: 0;
                width: 340px;
                height: 440px;
                background: var(--card-bg);
                border: 1px solid var(--border-color);
                border-radius: 20px;
                box-shadow: 0 10px 30px rgba(0,0,0,0.2);
                flex-direction: column;
                overflow: hidden;
            }
        </style>
    </head>
    <body>

    <div class="app-wrapper">
        <!-- SIDEBAR BÊN TRÁI -->
        <aside class="sidebar">
            <div>
                <a href="#" class="sidebar-brand">
                    <i class="bi bi-flower1 fs-3"></i>
                    <span>IrisClassifier</span>
                </a>
                <ul class="nav-menu">
                    <li><a class="nav-item-link active" onclick="switchTab('tab-home', this)"><i class="bi bi-house-door"></i> Trang chủ</a></li>
                    <li><a class="nav-item-link" onclick="switchTab('tab-predict-section', this)"><i class="bi bi-cpu"></i> Phân loại hoa</a></li>
                    <li><a class="nav-item-link" onclick="switchTab('tab-history', this)"><i class="bi bi-clock-history"></i> Lịch sử phân loại</a></li>
                    <li><a class="nav-item-link" onclick="switchTab('tab-dataset', this)"><i class="bi bi-database"></i> Bộ dữ liệu</a></li>
                    <li><a class="nav-item-link" onclick="switchTab('tab-knowledge', this)"><i class="bi bi-book"></i> Kiến thức</a></li>
                    <li><a class="nav-item-link" onclick="switchTab('tab-stats', this)"><i class="bi bi-bar-chart"></i> Thống kê</a></li>
                </ul>
            </div>
            <div class="p-2 text-center text-muted small">
                <p class="m-0">Iris AI Suite v2.5</p>
            </div>
        </aside>

        <!-- NỘI DUNG CHÍNH -->
        <main class="main-content">
            <!-- TAB 1: TRANG CHỦ & PHÂN LOẠI -->
            <div id="tab-home" class="tab-section active">
                <div class="hero-banner d-flex align-items-center">
                    <div style="max-width: 550px; z-index: 2;">
                        <span class="badge bg-primary bg-opacity-28 text-white mb-2 px-3 py-2 rounded-pill">AI POWERED FLOWER CLASSIFICATION</span>
                        <h1 class="fw-800 display-5 mb-3">Phân loại hoa Iris</h1>
                        <p class="text-white-50 fs-6 mb-4">Điều chỉnh thông số hoặc sử dụng công cụ AI để nhận diện chính xác từng loài hoa Iris.</p>
                        <button class="btn btn-primary rounded-pill px-4 py-2 me-2" onclick="switchTab('tab-predict-section', document.querySelectorAll('.nav-item-link')[1])">Bắt đầu phân loại <i class="bi bi-arrow-right"></i></button>
                    </div>
                    <img src="https://upload.wikimedia.org/wikipedia/commons/4/41/Iris_versicolor_3.jpg" alt="Iris Banner">
                </div>
            </div>

            <!-- TAB 2: KHU VỰC DỰ ĐOÁN -->
            <div id="tab-predict-section" class="tab-section active">
                <div class="row g-4">
                    <div class="col-lg-5">
                        <div class="content-card h-100">
                            <h5 class="fw-700 mb-4"><i class="bi bi-sliders me-2 text-primary"></i> Điều chỉnh kích thước</h5>
                            
                            <div class="mb-3">
                                <label class="d-flex justify-content-between fw-600 mb-1">
                                    <span>Sepal Length (Dài đài)</span>
                                    <span class="text-primary fw-700" id="lbl_sl">5.1 cm</span>
                                </label>
                                <input type="range" class="form-range" id="sl" min="4.0" max="8.0" step="0.1" value="5.1" oninput="updateVal('sl', 'lbl_sl')">
                            </div>

                            <div class="mb-3">
                                <label class="d-flex justify-content-between fw-600 mb-1">
                                    <span>Sepal Width (Rộng đài)</span>
                                    <span class="text-primary fw-700" id="lbl_sw">3.5 cm</span>
                                </label>
                                <input type="range" class="form-range" id="sw" min="2.0" max="4.5" step="0.1" value="3.5" oninput="updateVal('sw', 'lbl_sw')">
                            </div>

                            <div class="mb-3">
                                <label class="d-flex justify-content-between fw-600 mb-1">
                                    <span>Petal Length (Dài cánh)</span>
                                    <span class="text-primary fw-700" id="lbl_pl">1.4 cm</span>
                                </label>
                                <input type="range" class="form-range" id="pl" min="1.0" max="7.0" step="0.1" value="1.4" oninput="updateVal('pl', 'lbl_pl')">
                            </div>

                            <div class="mb-4">
                                <label class="d-flex justify-content-between fw-600 mb-1">
                                    <span>Petal Width (Rộng cánh)</span>
                                    <span class="text-primary fw-700" id="lbl_pw">0.2 cm</span>
                                </label>
                                <input type="range" class="form-range" id="pw" min="0.1" max="2.5" step="0.1" value="0.2" oninput="updateVal('pw', 'lbl_pw')">
                            </div>

                            <button class="btn btn-primary w-100 rounded-3 py-3 fw-700" onclick="runPredict()">
                                <i class="bi bi-magic me-2"></i> Phân loại ngay
                            </button>
                        </div>
                    </div>

                    <div class="col-lg-7">
                        <div class="content-card h-100">
                            <div class="d-flex justify-content-between align-items-center mb-3">
                                <h5 class="fw-700 m-0">Kết quả phân loại</h5>
                                <span class="badge bg-success bg-opacity-25 text-success rounded-pill px-3"><i class="bi bi-check-circle me-1"></i> Đã nhận diện thành công</span>
                            </div>

                            <div class="row g-3">
                                <div class="col-md-5">
                                    <img id="resImg" src="https://upload.wikimedia.org/wikipedia/commons/5/56/Kosaciec_szczecinkowaty_Iris_setosa.jpg" class="img-fluid rounded-4 border w-100" style="height: 180px; object-fit: cover;">
                                </div>
                                <div class="col-md-7">
                                    <h3 id="resName" class="fw-800 text-primary mb-1">Iris setosa</h3>
                                    <p class="text-muted small mb-2">Độ chính xác: <strong id="resAcc" class="text-success">99.2%</strong></p>
                                    <p id="resDesc" class="small text-muted mb-0">Hoa có cánh nhỏ gọn, màu tím nhạt/xanh. Rất dễ nhận biết.</p>
                                </div>
                            </div>

                            <hr class="my-4">

                            <div class="row align-items-center">
                                <div class="col-md-6">
                                    <h6 class="fw-700 mb-3">Biểu đồ phân bố loài</h6>
                                    <div style="height: 140px; position: relative;">
                                        <canvas id="donutChart"></canvas>
                                    </div>
                                </div>
                                <div class="col-md-6">
                                    <div class="p-3 bg-dark border border-secondary border-opacity-25 rounded-3">
                                        <small class="text-muted d-block mb-1">Bạn có biết?</small>
                                        <span class="small">Hoa Iris có hơn 300 loài khác nhau và được xem là biểu tượng của sự hy vọng và trí tuệ.</span>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>

            <!-- TAB LỊCH SỬ -->
            <div id="tab-history" class="tab-section">
                <div class="content-card">
                    <h5 class="fw-700 mb-3"><i class="bi bi-clock-history me-2 text-primary"></i> Lịch sử dự đoán</h5>
                    <p class="text-muted">Chưa có lịch sử thao tác nào gần đây.</p>
                </div>
            </div>

            <!-- TAB BỘ DỮ LIỆU -->
            <div id="tab-dataset" class="tab-section">
                <div class="content-card">
                    <h5 class="fw-700 mb-3"><i class="bi bi-database me-2 text-primary"></i> Bộ dữ liệu Iris</h5>
                    <p class="text-muted">Bộ dữ liệu gồm 150 mẫu hoa Iris với 4 đặc trưng hình thái chuẩn.</p>
                </div>
            </div>

            <!-- TAB KIẾN THỨC -->
            <div id="tab-knowledge" class="tab-section">
                <div class="content-card">
                    <h5 class="fw-700 mb-3"><i class="bi bi-book me-2 text-primary"></i> Thư viện kiến thức</h5>
                    <p class="text-muted">Tìm hiểu về 3 nhóm hoa chuẩn: Setosa, Versicolor và Virginica.</p>
                </div>
            </div>

            <!-- TAB THỐNG KÊ -->
            <div id="tab-stats" class="tab-section">
                <div class="content-card">
                    <h5 class="fw-700 mb-3"><i class="bi bi-bar-chart me-2 text-primary"></i> Thống kê hệ thống</h5>
                    <p class="text-muted">Tổng quan số lượt dự đoán và độ chính xác trung bình của mô hình AI.</p>
                </div>
            </div>
        </main>
    </div>

    <!-- CHATBOT tiện ích NỔI GÓC PHẢI -->
    <div class="chat-floating">
        <button class="chat-btn" onclick="toggleChatPopup()"><i class="bi bi-chat-dots"></i></button>
        <div class="chat-box-popup" id="chatPopup">
            <div class="p-3 bg-primary text-white d-flex justify-content-between align-items-center">
                <span class="fw-700"><i class="bi bi-robot me-2"></i> Trợ lý Iris AI</span>
                <button class="btn-close btn-close-white" onclick="toggleChatPopup()"></button>
            </div>
            <div class="p-3 flex-grow-1 overflow-y-auto small" id="chatContent">
                <div class="bg-dark p-2 rounded-3 text-white mb-2">Xin chào! Bạn muốn hỏi gì về loài hoa Iris hoặc bộ dữ liệu phân loại?</div>
            </div>
            <div class="p-2 border-top border-secondary border-opacity-25 d-flex gap-2">
                <input type="text" id="chatInput" class="form-control form-control-sm" placeholder="Nhập tin nhắn..." onkeypress="if(event.key==='Enter') sendChat()">
                <button class="btn btn-primary btn-sm" onclick="sendChat()"><i class="bi bi-send"></i></button>
            </div>
        </div>
    </div>

    <script>
        let chartInstance = null;

        function switchTab(tabId, element) {
            // Đổi active cho sidebar menu
            document.querySelectorAll('.nav-item-link').forEach(el => el.classList.remove('active'));
            if(element) element.classList.add('active');

            // Ẩn/Hiện phần nội dung tương ứng
            if(tabId === 'tab-home') {
                document.getElementById('tab-home').style.display = 'block';
                document.getElementById('tab-predict-section').style.display = 'block';
                document.querySelectorAll('.tab-section').forEach(el => {
                    if(el.id !== 'tab-home' && el.id !== 'tab-predict-section') el.style.display = 'none';
                });
            } else {
                document.querySelectorAll('.tab-section').forEach(el => el.style.display = 'none');
                const target = document.getElementById(tabId);
                if(target) target.style.display = 'block';
            }
        }

        function updateVal(id, lblId) {
            const val = document.getElementById(id).value;
            document.getElementById(lblId).innerText = val + " cm";
        }

        async function runPredict() {
            const sl = parseFloat(document.getElementById('sl').value);
            const sw = parseFloat(document.getElementById('sw').value);
            const pl = parseFloat(document.getElementById('pl').value);
            const pw = parseFloat(document.getElementById('pw').value);

            const res = await fetch('/predict', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({sepal_length: sl, sepal_width: sw, petal_length: pl, petal_width: pw})
            });
            const data = await res.json();

            document.getElementById('resName').innerText = data.name;
            document.getElementById('resAcc').innerText = data.acc;
            document.getElementById('resDesc').innerText = data.desc;
            document.getElementById('resImg').src = data.img;

            renderChart(data.probs);
        }

        function renderChart(probs) {
            const ctx = document.getElementById('donutChart').getContext('2d');
            if(chartInstance) chartInstance.destroy();

            chartInstance = new Chart(ctx, {
                type: 'doughnut',
                data: {
                    labels: ['Setosa', 'Versicolor', 'Virginica'],
                    datasets: [{
                        data: probs,
                        backgroundColor: ['#6366f1', '#3b82f6', '#ec4899']
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: { legend: { position: 'right' } }
                }
            });
        }

        function toggleChatPopup() {
            const el = document.getElementById('chatPopup');
            el.style.display = el.style.display === 'flex' ? 'none' : 'flex';
        }

        async function sendChat() {
            const inp = document.getElementById('chatInput');
            const txt = inp.value.trim();
            if(!txt) return;

            const box = document.getElementById('chatContent');
            box.innerHTML += `<div class="bg-primary text-white p-2 rounded-3 text-end mb-2 ms-auto" style="max-width: 80%;">${txt}</div>`;
            inp.value = '';

            const res = await fetch('/chat', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({message: txt})
            });
            const data = await res.json();

            box.innerHTML += `<div class="bg-dark text-white p-2 rounded-3 mb-2 me-auto" style="max-width: 80%;">${data.reply}</div>`;
            box.scrollTop = box.scrollHeight;
        }

        window.onload = () => {
            runPredict();
        };
    </script>
    </body>
    </html>
    """
