from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import joblib

# Nạp mô hình SVM đã huấn luyện
model = joblib.load("svm_model.pkl") 

app = FastAPI(title="Iris AI Neural Classifier")

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
        "origin": "Bắc Mỹ & Đông Bắc Á"
    },
    1: {
        "name": "IRIS VERSICOLOR",
        "badge": "Loài Phổ Biến - Nhóm 02",
        "img": "https://upload.wikimedia.org/wikipedia/commons/4/41/Iris_versicolor_3.jpg",
        "desc": "Kích thước trung bình, dải màu từ lam xám đến tím sẫm. Thuộc nhóm trung gian có đặc trưng biến thiên cao.",
        "habitat": "Ven sông, suối, vùng ven hồ",
        "origin": "Đông Bắc Bắc Mỹ"
    },
    2: {
        "name": "IRIS VIRGINICA",
        "badge": "Loài Kích Thước Lớn - Nhóm 03",
        "img": "https://upload.wikimedia.org/wikipedia/commons/9/9f/Iris_virginica.jpg",
        "desc": "Sở hữu cánh hoa và lá đài phát triển tối đa. Cấu trúc hình học vượt trội so với hai nhóm loài còn lại.",
        "habitat": "Đồng cỏ ẩm ướt, đầm lầy cạn",
        "origin": "Đông Nam Hoa Kỳ"
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
        <title>Iris AI Analytics - Core Engine</title>
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
        <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;600;800&display=swap" rel="stylesheet">
        <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
        <style>
            :root {
                --bg-gradient: linear-gradient(135deg, #0f172a 0%, #1e1b4b 50%, #0f172a 100%);
                --glass-bg: rgba(255, 255, 255, 0.05);
                --glass-border: rgba(255, 255, 255, 0.12);
                --accent-cyan: #38bdf8;
                --accent-purple: #c084fc;
            }

            body { 
                background: var(--bg-gradient); 
                color: #f8fafc; 
                min-height: 100vh; 
                font-family: 'Plus Jakarta Sans', sans-serif;
                overflow-x: hidden;
            }

            .card-main { 
                border-radius: 24px; 
                border: 1px solid var(--glass-border); 
                box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.5); 
                background: var(--glass-bg); 
                backdrop-filter: blur(16px); 
                position: relative;
                z-index: 1;
            }

            .text-gradient {
                background: linear-gradient(90deg, var(--accent-cyan), var(--accent-purple));
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
            }

            .badge-custom {
                background: rgba(56, 189, 248, 0.1);
                border: 1px solid rgba(56, 189, 248, 0.3);
                color: var(--accent-cyan);
                border-radius: 8px;
                padding: 6px 12px;
            }

            .preset-btn {
                background: rgba(255, 255, 255, 0.05);
                border: 1px solid var(--glass-border);
                color: #94a3b8;
                border-radius: 10px;
                padding: 6px 12px;
                font-size: 0.85rem;
                transition: all 0.2s;
            }
            .preset-btn:hover {
                background: rgba(56, 189, 248, 0.2);
                color: #fff;
                border-color: var(--accent-cyan);
            }

            .btn-predict { 
                background: linear-gradient(135deg, #6366f1 0%, #a855f7 100%); 
                color: #ffffff; 
                font-weight: 700; 
                border: none; 
                border-radius: 12px; 
                transition: all 0.3s;
                box-shadow: 0 4px 20px rgba(168, 85, 247, 0.4);
            }

            .btn-predict:hover { 
                transform: translateY(-2px); 
                box-shadow: 0 8px 30px rgba(168, 85, 247, 0.6); 
                color: #ffffff;
            }

            .custom-num-input {
                background: rgba(15, 23, 42, 0.6) !important;
                border: 1px solid var(--glass-border) !important;
                color: var(--accent-cyan) !important;
                font-weight: 700;
                text-align: center;
            }

            .btn-step {
                background: rgba(255, 255, 255, 0.08);
                border: 1px solid var(--glass-border);
                color: #fff;
                width: 42px;
            }
            .btn-step:hover {
                background: rgba(255, 255, 255, 0.2);
                color: #fff;
            }

            /* Khung ảnh nâng cấp hiệu ứng Hover & Neon Glow */
            .img-hover-box {
                position: relative;
                overflow: hidden;
                border-radius: 16px;
                border: 1px solid rgba(255, 255, 255, 0.15);
                box-shadow: 0 8px 24px rgba(0, 0, 0, 0.3);
                transition: all 0.4s ease-in-out;
            }

            .img-hover-box:hover {
                border-color: rgba(168, 85, 247, 0.6);
                box-shadow: 0 12px 32px rgba(168, 85, 247, 0.4);
                transform: translateY(-3px);
            }

            .flower-img { 
                width: 100%; 
                height: 220px; 
                object-fit: cover; 
                display: block;
                transition: transform 0.5s cubic-bezier(0.25, 1, 0.5, 1);
            }

            .img-hover-box:hover .flower-img {
                transform: scale(1.08);
            }

            .result-card { 
                display: none; 
                animation: slideUp 0.6s cubic-bezier(0.16, 1, 0.3, 1); 
            }

            @keyframes slideUp { 
                from { opacity: 0; transform: translateY(20px); } 
                to { opacity: 1; transform: translateY(0); } 
            }

            .radar-container {
                background: rgba(0, 0, 0, 0.2);
                border-radius: 16px;
                padding: 12px;
                border: 1px solid rgba(255, 255, 255, 0.05);
            }
        </style>
    </head>
    <body class="py-5">
        <div class="container">
            <div class="row justify-content-center">
                <div class="col-lg-10">
                    <div class="card card-main p-4 p-md-5">
                        
                        <!-- Header Section -->
                        <div class="text-center mb-4">
                            <h1 class="fw-800 text-gradient display-5">PHÂN LOẠI HOA IRIS 💐</h1>
                        </div>

                        <!-- Presets Section -->
                        <div class="mb-4 p-3 rounded-4 bg-black bg-opacity-20 border border-white border-opacity-10 text-center">
                            <span class="small text-secondary me-2">⚡Chọn nhanh loài hoa mẫu:</span>
                            <button class="preset-btn me-1" onclick="loadPreset(5.1, 3.5, 1.4, 0.2)">🌸 Setosa</button>
                            <button class="preset-btn me-1" onclick="loadPreset(6.0, 2.9, 4.5, 1.5)">🌺 Versicolor</button>
                            <button class="preset-btn" onclick="loadPreset(6.5, 3.0, 5.5, 2.0)">🌻 Virginica</button>
                        </div>
                        
                        <div class="row g-4">
                            <!-- Input Column -->
                            <div class="col-md-6 border-end border-secondary border-opacity-25 pe-md-4">
                                <h5 class="fw-600 mb-4 text-light d-flex align-items-center gap-2">
                                    <span>🎛️</span> Thông số đầu vào (cm)
                                </h5>
                                <form id="irisForm">
                                    
                                    <!-- Field 1 -->
                                    <div class="mb-3">
                                        <label class="form-label text-secondary small mb-1">Chiều dài lá đài (Sepal Length)</label>
                                        <div class="input-group">
                                            <button class="btn btn-step" type="button" onclick="stepVal('sepal_length', -0.1)">-</button>
                                            <input type="number" class="form-control custom-num-input" id="sepal_length" value="5.1" step="0.1" min="1" max="10">
                                            <button class="btn btn-step" type="button" onclick="stepVal('sepal_length', 0.1)">+</button>
                                        </div>
                                    </div>

                                    <!-- Field 2 -->
                                    <div class="mb-3">
                                        <label class="form-label text-secondary small mb-1">Chiều rộng lá đài (Sepal Width)</label>
                                        <div class="input-group">
                                            <button class="btn btn-step" type="button" onclick="stepVal('sepal_width', -0.1)">-</button>
                                            <input type="number" class="form-control custom-num-input" id="sepal_width" value="3.5" step="0.1" min="1" max="10">
                                            <button class="btn btn-step" type="button" onclick="stepVal('sepal_width', 0.1)">+</button>
                                        </div>
                                    </div>

                                    <!-- Field 3 -->
                                    <div class="mb-3">
                                        <label class="form-label text-secondary small mb-1">Chiều dài cánh hoa (Petal Length)</label>
                                        <div class="input-group">
                                            <button class="btn btn-step" type="button" onclick="stepVal('petal_length', -0.1)">-</button>
                                            <input type="number" class="form-control custom-num-input" id="petal_length" value="1.4" step="0.1" min="1" max="10">
                                            <button class="btn btn-step" type="button" onclick="stepVal('petal_length', 0.1)">+</button>
                                        </div>
                                    </div>

                                    <!-- Field 4 -->
                                    <div class="mb-4">
                                        <label class="form-label text-secondary small mb-1">Chiều rộng cánh hoa (Petal Width)</label>
                                        <div class="input-group">
                                            <button class="btn btn-step" type="button" onclick="stepVal('petal_width', -0.1)">-</button>
                                            <input type="number" class="form-control custom-num-input" id="petal_width" value="0.2" step="0.1" min="0.1" max="10">
                                            <button class="btn btn-step" type="button" onclick="stepVal('petal_width', 0.1)">+</button>
                                        </div>
                                    </div>
                                    
                                    <button type="button" class="btn btn-predict w-100 py-3 mt-2 fs-6" onclick="makePrediction()">
                                        🔮 PHÂN LOẠI NGAY
                                    </button>
                                </form>
                            </div>

                            <!-- Output Column -->
                            <div class="col-md-6 d-flex flex-column justify-content-center ps-md-4">
                                
                                <!-- Standby Placeholder -->
                                <div id="placeholderText" class="text-center text-secondary my-auto py-5">
                                    <div class="mb-3 fs-1 opacity-50">📡</div>
                                    <h6 class="fw-600 text-light">Đang chờ tín hiệu dữ liệu...</h6>
                                    <p class="small text-muted mb-0">Chọn mẫu nhanh hoặc tự điều chỉnh thông số bên trái để phân tích.</p>
                                </div>

                                <!-- Result Visualizer -->
                                <div id="resultCard" class="result-card">
                                    <div class="d-flex align-items-center justify-content-between mb-3">
                                        <div>
                                            <span id="flowerBadge" class="badge bg-primary bg-opacity-25 text-info border border-info border-opacity-25 mb-1"></span>
                                            <h3 id="flowerName" class="fw-800 text-light m-0"></h3>
                                        </div>
                                        <span class="badge bg-success bg-opacity-25 text-success border border-success border-opacity-25">Matched 100%</span>
                                    </div>

                                    <!-- Khung ảnh có hiệu ứng Hover -->
                                    <div class="img-hover-box mb-3">
                                        <img id="flowerImg" src="" class="flower-img" alt="Predicted Specimen">
                                    </div>
                                    
                                    <p id="flowerDesc" class="text-secondary small mb-2"></p>

                                    <!-- Thẻ chỉ số sinh học chi tiết bổ sung -->
                                    <div class="row g-2 mb-3 text-start">
                                        <div class="col-6">
                                            <div class="p-2 rounded-3" style="background: rgba(255,255,255,0.05); border: 1px solid rgba(255,255,255,0.1);">
                                                <small class="text-secondary d-block" style="font-size: 0.72rem;">🏡 Môi trường sống</small>
                                                <span class="fw-600 text-light" style="font-size: 0.8rem;" id="habitatVal">--</span>
                                            </div>
                                        </div>
                                        <div class="col-6">
                                            <div class="p-2 rounded-3" style="background: rgba(255,255,255,0.05); border: 1px solid rgba(255,255,255,0.1);">
                                                <small class="text-secondary d-block" style="font-size: 0.72rem;">📍 Phân bố chính</small>
                                                <span class="fw-600 text-light" style="font-size: 0.8rem;" id="originVal">--</span>
                                            </div>
                                        </div>
                                    </div>

                                    <!-- Radar Chart Section -->
                                    <div class="radar-container" style="height: 180px;">
                                        <canvas id="radarChart"></canvas>
                                    </div>
                                </div>

                            </div>
                        </div>

                    </div>
                </div>
            </div>
        </div>

        <!-- Script hiệu ứng hạt neon & logic ứng dụng -->
        <script>
            // Canvas Particle Background (Hạt neon chuyển động)
            const canvas = document.createElement('canvas');
            canvas.style.position = 'fixed';
            canvas.style.top = '0';
            canvas.style.left = '0';
            canvas.style.width = '100%';
            canvas.style.height = '100%';
            canvas.style.pointerEvents = 'none';
            canvas.style.zIndex = '0';
            document.body.appendChild(canvas);

            const ctx = canvas.getContext('2d');
            let particles = [];

            function resize() {
                canvas.width = window.innerWidth;
                canvas.height = window.innerHeight;
            }
            window.addEventListener('resize', resize);
            resize();

            for(let i = 0; i < 35; i++) {
                particles.push({
                    x: Math.random() * canvas.width,
                    y: Math.random() * canvas.height,
                    r: Math.random() * 2 + 1,
                    dx: (Math.random() - 0.5) * 0.4,
                    dy: (Math.random() - 0.5) * 0.4,
                    alpha: Math.random() * 0.5 + 0.2
                });
            }

            function animateParticles() {
                ctx.clearRect(0, 0, canvas.width, canvas.height);
                particles.forEach(p => {
                    p.x += p.dx;
                    p.y += p.dy;
                    if(p.x < 0 || p.x > canvas.width) p.dx *= -1;
                    if(p.y < 0 || p.y > canvas.height) p.dy *= -1;
                    ctx.beginPath();
                    ctx.arc(p.x, p.y, p.r, 0, Math.PI * 2);
                    ctx.fillStyle = `rgba(168, 85, 247, ${p.alpha})`;
                    ctx.fill();
                });
                requestAnimationFrame(animateParticles);
            }
            animateParticles();

            // Core Web Logic
            let radarChart = null;

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
                document.getElementById('resultCard').style.display = 'block';

                document.getElementById('flowerName').innerText = result.info.name;
                document.getElementById('flowerBadge').innerText = result.info.badge;
                document.getElementById('flowerImg').src = result.info.img;
                document.getElementById('flowerDesc').innerText = result.info.desc;
                document.getElementById('habitatVal').innerText = result.info.habitat;
                document.getElementById('originVal').innerText = result.info.origin;

                renderRadar([sl, sw, pl, pw]);
            }

            function renderRadar(inputData) {
                const ctxRadar = document.getElementById('radarChart').getContext('2d');
                
                if (radarChart) { radarChart.destroy(); }

                radarChart = new Chart(ctxRadar, {
                    type: 'radar',
                    data: {
                        labels: ['Sepal Length', 'Sepal Width', 'Petal Length', 'Petal Width'],
                        datasets: [{
                            label: 'Vector chỉ số',
                            data: inputData,
                            backgroundColor: 'rgba(56, 189, 248, 0.25)',
                            borderColor: '#38bdf8',
                            pointBackgroundColor: '#c084fc',
                            pointBorderColor: '#ffffff',
                            borderWidth: 2
                        }]
                    },
                    options: {
                        responsive: true,
                        maintainAspectRatio: false,
                        plugins: { legend: { display: false } },
                        scales: { 
                            r: { 
                                min: 0, 
                                max: 8,
                                grid: { color: 'rgba(255, 255, 255, 0.1)' },
                                angleLines: { color: 'rgba(255, 255, 255, 0.1)' },
                                pointLabels: { color: '#94a3b8', font: { size: 10 } },
                                ticks: { display: false }
                            } 
                        }
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
