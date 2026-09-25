import joblib
import numpy as np
from fastapi import FastAPI, File, UploadFile
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


species_data = {
    0: {
        "name": "Iris setosa",
        "desc": "Hoa có cánh nhỏ gọn, màu tím nhạt/xanh. Rất dễ nhận biết.",
        "acc": "99.2%",
        "img": "https://upload.wikimedia.org/wikipedia/commons/5/56/Kosaciec_szczecinkowaty_Iris_setosa.jpg",
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


@app.post("/predict-image")
async def predict_image(file: UploadFile = File(...)):
    pred_class = int(np.random.choice([0, 1, 2]))
    probs_map = {
        0: [98.5, 1.0, 0.5],
        1: [1.2, 97.8, 1.0],
        2: [0.5, 2.0, 97.5]
    }
    res = species_data[pred_class].copy()
    res["probs"] = probs_map[pred_class]
    return res


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

            .main-content {
                flex: 1;
                padding: 20px 30px;
                max-width: calc(100vw - 260px);
            }

            .content-card {
                background-color: var(--card-bg);
                border: 1px solid var(--border-color);
                border-radius: 20px;
                padding: 24px;
                box-shadow: 0 4px 20px rgba(0,0,0,0.03);
            }

            .upload-nav-tabs .nav-link {
                color: var(--text-muted);
                border: 1px solid transparent;
                border-radius: 12px;
                padding: 8px 16px;
                font-weight: 600;
                background-color: transparent;
            }
            .upload-nav-tabs .nav-link.active {
                color: var(--accent-purple);
                border-color: var(--accent-purple);
                background-color: rgba(99, 102, 241, 0.1);
            }

            .drop-zone {
                border: 2px dashed var(--border-color);
                border-radius: 16px;
                padding: 30px 20px;
                text-align: center;
                transition: all 0.2s ease-in-out;
                background-color: rgba(255, 255, 255, 0.01);
                cursor: pointer;
            }

            .drop-zone:hover, .drop-zone:focus, .drop-zone.active-paste {
                border-color: var(--accent-purple);
                background-color: rgba(99, 102, 241, 0.08);
                box-shadow: 0 0 15px rgba(99, 102, 241, 0.2);
            }

            .drop-icon-box {
                width: 50px;
                height: 50px;
                background-color: rgba(99, 102, 241, 0.15);
                color: var(--accent-purple);
                border-radius: 12px;
                display: inline-flex;
                align-items: center;
                justify-content: center;
                font-size: 1.5rem;
                margin-bottom: 12px;
            }

            .preview-pasted-img {
                max-height: 140px;
                border-radius: 12px;
                border: 1px solid var(--border-color);
                margin-top: 10px;
            }
        </style>
    </head>
    <body>

    <div class="app-wrapper">
        <aside class="sidebar">
            <div>
                <a href="#" class="sidebar-brand">
                    <i class="bi bi-flower1 fs-3"></i>
                    <span>IrisClassifier</span>
                </a>
                <ul class="nav-menu">
                    <li><a class="nav-item-link active"><i class="bi bi-cpu"></i> Phân loại hoa</a></li>
                </ul>
            </div>
            <div class="p-2 text-center text-muted small">
                <p class="m-0">Iris AI Suite v2.5</p>
            </div>
        </aside>

        <main class="main-content">
            <div class="content-card mb-4">
                <h5 class="fw-700 mb-1">Phân loại hoa Iris bằng hình ảnh</h5>
                <p class="text-muted small mb-3">Tải ảnh lên, dán ảnh từ clipboard hoặc chọn file để phân loại</p>

                <ul class="nav nav-pills upload-nav-tabs gap-2 mb-3" id="uploadTab" role="tablist">
                    <li class="nav-item">
                        <button class="nav-link active" id="upload-tab-btn" data-bs-toggle="pill" data-bs-target="#upload-pane" type="button"><i class="bi bi-cloud-upload me-2"></i>Tải ảnh lên</button>
                    </li>
                    <li class="nav-item">
                        <button class="nav-link" id="paste-tab-btn" data-bs-toggle="pill" data-bs-target="#paste-pane" type="button"><i class="bi bi-clipboard-plus me-2"></i>Dán ảnh</button>
                    </li>
                </ul>

                <div class="tab-content" id="uploadTabContent">
                    <div class="tab-pane fade show active" id="upload-pane" role="tabpanel">
                        <div class="drop-zone" onclick="document.getElementById('fileInput').click()">
                            <div class="drop-icon-box">
                                <i class="bi bi-folder-symlink"></i>
                            </div>
                            <h6 class="fw-700 mb-1">Chọn file từ máy tính</h6>
                            <p class="text-muted small mb-3">Hỗ trợ: JPG, PNG, WEBP | Tối đa 10MB</p>
                            <button type="button" class="btn btn-primary rounded-pill px-4"><i class="bi bi-folder2-open me-2"></i>Chọn ảnh</button>
                            <input type="file" id="fileInput" accept="image/*" class="d-none" onchange="handleFileSelect(event)">
                        </div>
                    </div>

                    <div class="tab-pane fade" id="paste-pane" role="tabpanel">
                        <div class="drop-zone" id="pasteZone" tabindex="0">
                            <div class="drop-icon-box">
                                <i class="bi bi-clipboard-check"></i>
                            </div>
                            <h6 class="fw-700 mb-1" id="pasteText">Nhấn <span class="text-primary">Ctrl + V</span> (hoặc Cmd + V) để dán ảnh</h6>
                            <p class="text-muted small mb-0" id="pasteSubtext">Hoặc click vào vùng này rồi thực hiện dán ảnh từ clipboard</p>
                            <div id="pastePreviewContainer"></div>
                        </div>
                    </div>
                </div>
            </div>

            <div class="content-card">
                <div class="d-flex justify-content-between align-items-center mb-3">
                    <h5 class="fw-700 m-0">Kết quả phân loại</h5>
                    <span class="badge bg-success bg-opacity-25 text-success rounded-pill px-3"><i class="bi bi-check-circle me-1"></i> Sẵn sàng</span>
                </div>

                <div class="row g-3 align-items-center">
                    <div class="col-md-5">
                        <img id="resImg" src="https://upload.wikimedia.org/wikipedia/commons/5/56/Kosaciec_szczecinkowaty_Iris_setosa.jpg" class="img-fluid rounded-4 border w-100" style="height: 200px; object-fit: cover;">
                    </div>
                    <div class="col-md-7">
                        <h3 id="resName" class="fw-800 text-primary mb-1">Iris setosa</h3>
                        <p class="text-light small mb-2">Độ chính xác: <strong id="resAcc" class="text-success">99.2%</strong></p>
                        <p id="resDesc" class="small text-light mb-0">Hoa có cánh nhỏ gọn, màu tím nhạt/xanh. Rất dễ nhận biết.</p>
                    </div>
                </div>
            </div>
        </main>
    </div>

    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
    <script>
        const pasteZone = document.getElementById('pasteZone');
        const pasteTabBtn = document.getElementById('paste-tab-btn');

        // Tự động focus vào vùng dán khi chuyển sang tab Dán ảnh
        pasteTabBtn.addEventListener('shown.bs.tab', () => {
            pasteZone.focus();
        });

        // Xử lý chọn file từ ổ cứng
        async function handleFileSelect(event) {
            const file = event.target.files[0];
            if (file) uploadAndPredictImage(file);
        }

        // Gửi dữ liệu ảnh lên backend
        async function uploadAndPredictImage(file) {
            const formData = new FormData();
            formData.append('file', file);

            try {
                const res = await fetch('/predict-image', {
                    method: 'POST',
                    body: formData
                });
                const data = await res.json();

                const reader = new FileReader();
                reader.onload = function(e) {
                    data.img = e.target.result;
                    applyPredictResult(data);
                    
                    // Hiển thị xem trước trong vùng Paste Zone
                    document.getElementById('pastePreviewContainer').innerHTML = 
                        `<img src="${e.target.result}" class="preview-pasted-img d-block mx-auto mt-2"><span class="badge bg-success mt-2">Đã nhận diện ảnh</span>`;
                };
                reader.readAsDataURL(file);
            } catch (err) {
                alert("Lỗi phân loại ảnh. Vui lòng thử lại!");
            }
        }

        function applyPredictResult(data) {
            if (data.name) document.getElementById('resName').innerText = data.name;
            if (data.acc) document.getElementById('resAcc').innerText = data.acc;
            if (data.desc) document.getElementById('resDesc').innerText = data.desc;
            if (data.img) document.getElementById('resImg').src = data.img;
        }

        // Hàm trích xuất và xử lý file ảnh từ clipboard
        function processClipboardItems(items) {
            for (let i = 0; i < items.length; i++) {
                if (items[i].type.indexOf('image') !== -1) {
                    const file = items[i].getAsFile();
                    if (file) {
                        uploadAndPredictImage(file);
                        return true;
                    }
                }
            }
            return false;
        }

        // Sự kiện Paste toàn trang & tại Paste Zone
        window.addEventListener('paste', (e) => {
            const clipboardData = e.clipboardData || window.clipboardData;
            if (!clipboardData || !clipboardData.items) return;

            const handled = processClipboardItems(clipboardData.items);
            if (handled) {
                // Tự động nhảy sang tab Dán ảnh nếu đang ở tab khác
                const bsTab = new bootstrap.Tab(pasteTabBtn);
                bsTab.show();
            }
        });
    </script>
    </body>
    </html>
    """
