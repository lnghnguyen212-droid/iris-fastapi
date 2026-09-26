import io
import base64
from fastapi import FastAPI, UploadFile, File, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image

app = FastAPI(title="Iris Classifier Pro")

# --- MÔ PHỎNG DỰ ĐOÁN VÀ DỮ LIỆU ---
DESCRIPTIONS = {
    "Iris-setosa": "Hoa Linh Lan San (Setosa) có đài hoa rộng, thường mọc ở các vùng khí hậu lạnh. Dễ phân biệt nhất trong các loài Iris.",
    "Iris-versicolor": "Hoa Diên Vĩ Đa Sắc (Versicolor) có kích thước trung bình, màu sắc sặc sỡ và ranh giới phân tách phức tạp hơn.",
    "Iris-virginica": "Hoa Diên Vĩ Virginia (Virginica) có cánh hoa lớn nhất trong 3 loài, thường xuất hiện ở khu vực ẩm ướt."
}

SAMPLE_IMAGES = {
    "Iris-setosa": "https://upload.wikimedia.org/wikipedia/commons/a/a7/Iris_setosa_iris.jpg",
    "Iris-versicolor": "https://upload.wikimedia.org/wikipedia/commons/4/41/Iris_versicolor_3.jpg",
    "Iris-virginica": "https://upload.wikimedia.org/wikipedia/commons/9/9f/Iris_virginica.jpg"
}

@app.post("/predict")
async def predict(
    sepal_length: float = Form(None),
    sepal_width: float = Form(None),
    petal_length: float = Form(None),
    petal_width: float = Form(None),
    kernel: str = Form("linear"),
    file: UploadFile = File(None)
):
    # Logic dự đoán mẫu (Mô phỏng trả về kết quả)
    predicted_species = "Iris-setosa"
    probs = [95.2, 3.8, 1.0]

    if petal_length and petal_length > 2.5:
        if petal_width and petal_width > 1.7:
            predicted_species = "Iris-virginica"
            probs = [0.5, 4.5, 95.0]
        else:
            predicted_species = "Iris-versicolor"
            probs = [1.2, 92.8, 6.0]

    return {
        "name": predicted_species,
        "desc": DESCRIPTIONS[predicted_species],
        "img": SAMPLE_IMAGES[predicted_species],
        "used_kernel": kernel.upper(),
        "probs": probs
    }

@app.get("/", response_class=HTMLResponse)
async def read_item():
    return """
<!DOCTYPE html>
<html lang="vi">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Iris Classifier Pro Dashboard</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <link href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.10.0/font/bootstrap-icons.css" rel="stylesheet">
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        body { background-color: #0f172a; color: #f8fafc; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; }
        .card { background-color: #1e293b; border: 1px solid #334155; border-radius: 12px; color: #f8fafc; }
        .nav-tabs .nav-link { color: #94a3b8; border: none; }
        .nav-tabs .nav-link.active { color: #38bdf8; background-color: transparent; border-bottom: 2px solid #38bdf8; }
        .form-control, .form-select { background-color: #0f172a; border: 1px solid #334155; color: #f8fafc; }
        .form-control:focus, .form-select:focus { background-color: #0f172a; color: #f8fafc; border-color: #38bdf8; box-shadow: none; }
        .progress { background-color: #334155; height: 10px; border-radius: 5px; }
        #pasteZone { border: 2px dashed #475569; border-radius: 8px; padding: 30px; text-align: center; cursor: pointer; transition: 0.3s; }
        #pasteZone:hover { border-color: #38bdf8; background-color: rgba(56, 189, 248, 0.05); }
    </style>
</head>
<body class="py-4">
    <div class="container">
        <header class="pb-3 mb-4 border-bottom border-secondary d-flex justify-content-between align-items-center">
            <h1 class="h3 text-info"><i class="bi bi-flower1 me-2"></i>Iris Classifier Pro Dashboard</h1>
            <span class="badge bg-primary fs-6">FastAPI + SVM</span>
        </header>

        <div class="row g-4">
            <!-- Cột trái: Nhập liệu & Tham số -->
            <div class="col-lg-5">
                <div class="card p-3 shadow-sm h-100">
                    <ul class="nav nav-tabs mb-3" id="inputTab" role="tablist">
                        <li class="nav-item">
                            <button class="nav-link active" id="manual-tab" data-bs-toggle="tab" data-bs-target="#manual" type="button"><i class="bi bi-sliders me-1"></i>Nhập Thông Số</button>
                        </li>
                        <li class="nav-item">
                            <button class="nav-link" id="paste-tab-btn" data-bs-toggle="tab" data-bs-target="#paste" type="button"><i class="bi bi-clipboard-plus me-1"></i>Dán Ảnh</button>
                        </li>
                    </ul>

                    <div class="tab-content" id="inputTabContent">
                        <!-- Tab 1: Nhập thông số -->
                        <div class="tab-pane fade show active" id="manual">
                            <form id="predictForm" onsubmit="handleFormSubmit(event)">
                                <div class="mb-3">
                                    <label class="form-label">Chọn SVM Kernel:</label>
                                    <select class="form-select" id="kernelSelect" onchange="renderBoundaryChart(this.value)">
                                        <option value="linear" selected>Linear (Tuyến tính)</option>
                                        <option value="rbf">RBF (Radial Basis Function)</option>
                                        <option value="poly">Polynomial (Đa thức)</option>
                                        <option value="sigmoid">Sigmoid</option>
                                    </select>
                                </div>
                                <div class="row g-2 mb-2">
                                    <div class="col-6">
                                        <label class="form-label small">Sepal Length (cm)</label>
                                        <input type="number" step="0.1" class="form-control" id="sl" value="5.1" required>
                                    </div>
                                    <div class="col-6">
                                        <label class="form-label small">Sepal Width (cm)</label>
                                        <input type="number" step="0.1" class="form-control" id="sw" value="3.5" required>
                                    </div>
                                </div>
                                <div class="row g-2 mb-3">
                                    <div class="col-6">
                                        <label class="form-label small">Petal Length (cm)</label>
                                        <input type="number" step="0.1" class="form-control" id="pl" value="1.4" required>
                                    </div>
                                    <div class="col-6">
                                        <label class="form-label small">Petal Width (cm)</label>
                                        <input type="number" step="0.1" class="form-control" id="pw" value="0.2" required>
                                    </div>
                                </div>
                                <button type="submit" class="btn btn-info w-100 fw-bold"><i class="bi bi-cpu me-1"></i> Dự Đoán Ngay</button>
                            </form>
                        </div>

                        <!-- Tab 2: Dán ảnh -->
                        <div class="tab-pane fade" id="paste">
                            <div id="pasteZone" tabindex="0">
                                <i class="bi bi-cloud-arrow-up display-4 text-info"></i>
                                <p class="mt-2 mb-0">Nhấp vào đây và ấn <strong>Ctrl + V</strong> để dán ảnh hoa Iris</p>
                            </div>
                        </div>
                    </div>
                </div>
            </div>

            <!-- Cột giữa: Kết quả dự đoán -->
            <div class="col-lg-7">
                <div class="card p-3 shadow-sm h-100">
                    <h5 class="card-title text-info mb-3"><i class="bi bi-check-circle me-2"></i>Kết Quả Phân Loại</h5>
                    <div class="row align-items-center">
                        <div class="col-md-5 text-center mb-3 mb-md-0">
                            <img id="resImg" src="https://upload.wikimedia.org/wikipedia/commons/a/a7/Iris_setosa_iris.jpg" class="img-fluid rounded shadow" style="max-height: 180px; object-fit: cover;" alt="Iris">
                        </div>
                        <div class="col-md-7">
                            <h3 id="resName" class="text-warning mb-1">Iris-setosa</h3>
                            <span id="usedKernelBadge" class="badge bg-secondary mb-2"><i class="bi bi-gear-wide-connected me-1"></i> Kernel: LINEAR</span>
                            <p id="resDesc" class="small text-light opacity-75">Hoa Linh Lan San (Setosa) có đài hoa rộng, thường mọc ở các vùng khí hậu lạnh.</p>

                            <!-- Xác suất -->
                            <div class="mt-3">
                                <div class="d-flex justify-content-between small"><span>Setosa</span><span id="prob_0">95.2%</span></div>
                                <div class="progress mb-2"><div id="bar_0" class="progress-bar bg-info" style="width: 95.2%"></div></div>

                                <div class="d-flex justify-content-between small"><span>Versicolor</span><span id="prob_1">3.8%</span></div>
                                <div class="progress mb-2"><div id="bar_1" class="progress-bar bg-warning" style="width: 3.8%"></div></div>

                                <div class="d-flex justify-content-between small"><span>Virginica</span><span id="prob_2">1.0%</span></div>
                                <div class="progress"><div id="bar_2" class="progress-bar bg-danger" style="width: 1.0%"></div></div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>

            <!-- Biểu đồ & Lịch sử -->
            <div class="col-lg-6">
                <div class="card p-3 shadow-sm">
                    <h6 class="text-info"><i class="bi bi-diagram-3 me-2"></i>Ranh Giới Phân Tách Kernel (Mô Phỏng)</h6>
                    <div style="height: 220px;"><canvas id="boundaryChart"></canvas></div>
                </div>
            </div>

            <div class="col-lg-6">
                <div class="card p-3 shadow-sm">
                    <div class="d-flex justify-content-between align-items-center mb-2">
                        <h6 class="text-info mb-0"><i class="bi bi-clock-history me-2"></i>Lịch Sử Dự Đoán</h6>
                        <button class="btn btn-sm btn-outline-danger" onclick="clearHistory()">Xóa</button>
                    </div>
                    <div class="table-responsive" style="max-height: 180px;">
                        <table class="table table-dark table-hover table-sm small">
                            <thead>
                                <tr><th>Thời gian</th><th>Kernel</th><th>Kết quả</th></tr>
                            </thead>
                            <tbody id="historyTableBody">
                                <!-- Dữ liệu lịch sử sẽ thêm vào đây -->
                            </tbody>
                        </table>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script me-2 src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
    <script>
        let boundaryChartInstance = null;
        let barChartInstance = null;
        let historyLogs = [];

        // Xử lý Form Submit
        async function handleFormSubmit(e) {
            e.preventDefault();
            const formData = new FormData();
            formData.append('sepal_length', document.getElementById('sl').value);
            formData.append('sepal_width', document.getElementById('sw').value);
            formData.append('petal_length', document.getElementById('pl').value);
            formData.append('petal_width', document.getElementById('pw').value);
            formData.append('kernel', document.getElementById('kernelSelect').value);

            try {
                const res = await fetch('/predict', { method: 'POST', body: formData });
                const data = await res.json();
                applyPredictResult(data);
            } catch (err) {
                console.error("Lỗi dự đoán:", err);
            }
        }

        // Upload và phân loại từ Clipboard File
        async function uploadAndPredictImage(file) {
            const formData = new FormData();
            formData.append('file', file);
            formData.append('kernel', document.getElementById('kernelSelect').value);

            try {
                const res = await fetch('/predict', { method: 'POST', body: formData });
                const data = await res.json();
                applyPredictResult(data);
            } catch (err) {
                console.error("Lỗi tải ảnh:", err);
            }
        }

        // Lắng nghe sự kiện Ctrl + V
        document.addEventListener('DOMContentLoaded', () => {
            const pasteZone = document.getElementById('pasteZone');
            if (pasteZone) {
                pasteZone.addEventListener('paste', (e) => {
                    const items = (e.clipboardData || e.originalEvent.clipboardData).items;
                    for (let i = 0; i < items.length; i++) {
                        if (items[i].type.indexOf('image') !== -1) {
                            const file = items[i].getAsFile();
                            uploadAndPredictImage(file);
                            break;
                        }
                    }
                });
            }
            renderBoundaryChart('linear');
        });

        // Hiển thị kết quả lên màn hình
        function applyPredictResult(data) {
            if (!data) return;
            document.getElementById('resName').innerText = data.name || 'N/A';
            document.getElementById('resDesc').innerText = data.desc || '';
            document.getElementById('resImg').src = data.img || '';
            
            const badge = document.getElementById('usedKernelBadge');
            if (badge) badge.innerHTML = `<i class="bi bi-gear-wide-connected me-1"></i> Kernel: ${data.used_kernel || 'N/A'}`;

            if (data.probs && data.probs.length === 3) {
                document.getElementById('prob_0').innerText = data.probs[0] + "%";
                document.getElementById('bar_0').style.width = data.probs[0] + "%";
                document.getElementById('prob_1').innerText = data.probs[1] + "%";
                document.getElementById('bar_1').style.width = data.probs[1] + "%";
                document.getElementById('prob_2').innerText = data.probs[2] + "%";
                document.getElementById('bar_2').style.width = data.probs[2] + "%";
            }

            addHistoryLog(data.used_kernel, data.name);
        }

        function addHistoryLog(kernel, name) {
            const timeStr = new Date().toLocaleTimeString();
            historyLogs.unshift({ time: timeStr, kernel: kernel || 'N/A', result: name || 'N/A' });
            updateHistoryTable();
        }

        function updateHistoryTable() {
            const tbody = document.getElementById('historyTableBody');
            if (!tbody) return;
            tbody.innerHTML = historyLogs.map(item => `
                <tr>
                    <td>${item.time}</td>
                    <td><span class="badge bg-secondary">${item.kernel}</span></td>
                    <td><span class="badge bg-primary">${item.result}</span></td>
                </tr>
            `).join('');
        }

        function clearHistory() {
            historyLogs = [];
            updateHistoryTable();
        }

        // Biểu đồ ranh giới
        function renderBoundaryChart(kernel) {
            const ctx = document.getElementById('boundaryChart');
            if (!ctx) return;

            if (boundaryChartInstance) boundaryChartInstance.destroy();

            let curveData = [{x: 1, y: 1}, {x: 3, y: 2.5}, {x: 5, y: 4}, {x: 7, y: 5.5}];
            if (kernel === 'rbf') curveData = [{x: 1, y: 2}, {x: 2.5, y: 4.5}, {x: 4.5, y: 4.5}, {x: 6, y: 2}];
            else if (kernel === 'poly') curveData = [{x: 1, y: 1}, {x: 2.5, y: 3.5}, {x: 4, y: 2}, {x: 6.5, y: 5}];

            boundaryChartInstance = new Chart(ctx, {
                type: 'scatter',
                data: {
                    datasets: [
                        {
                            label: 'Ranh giới (' + kernel.toUpperCase() + ')',
                            data: curveData,
                            showLine: true,
                            borderColor: '#38bdf8',
                            backgroundColor: 'rgba(56, 189, 248, 0.2)',
                            tension: 0.4,
                            pointRadius: 0
                        },
                        { label: 'Setosa', data: [{x: 1.4, y: 0.2}, {x: 1.3, y: 0.2}], backgroundColor: '#38bdf8' },
                        { label: 'Versicolor', data: [{x: 4.5, y: 1.5}, {x: 4.0, y: 1.3}], backgroundColor: '#f59e0b' },
                        { label: 'Virginica', data: [{x: 6.0, y: 2.5}, {x: 5.1, y: 1.9}], backgroundColor: '#ef4444' }
                    ]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {
                        x: { grid: { color: '#334155' }, ticks: { color: '#94a3b8' } },
                        y: { grid: { color: '#334155' }, ticks: { color: '#94a3b8' } }
                    },
                    plugins: { legend: { labels: { color: '#f8fafc', font: { size: 10 } } } }
                }
            });
        }
    </script>
</body>
</html>
    """

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
