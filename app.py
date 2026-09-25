import io
import numpy as np
from PIL import Image, ImageFilter
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import HTMLResponse

app = FastAPI()

# 1. Giao diện HTML được nhúng trực tiếp
HTML_CONTENT = """
<!DOCTYPE html>
<html lang="vi">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Dự Đoán & Xử Lý Ảnh</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <style>
        .drop-zone--over { border-color: #3b82f6; background-color: #eff6ff; }
    </style>
</head>
<body class="bg-slate-50 text-slate-800 min-h-screen p-6">

    <div class="max-w-4xl mx-auto bg-white shadow-xl rounded-2xl p-8">
        <h1 class="text-2xl font-bold text-center text-slate-900 mb-6">Hệ Thống Dự Đoán & Lọc Ảnh</h1>

        <!-- Khu vực chọn Kernel -->
        <div class="mb-6">
            <label for="kernelSelect" class="block font-semibold text-slate-700 mb-2">Chọn Kernel (Bộ lọc ảnh):</label>
            <select id="kernelSelect" class="w-full p-3 border border-slate-300 rounded-xl focus:ring-2 focus:ring-blue-500 focus:outline-none">
                <option value="none">Không dùng Kernel (Mặc định)</option>
                <option value="gaussian">Gaussian Blur (Làm mịn / Giảm nhiễu)</option>
                <option value="sobel">Sobel / Prewitt (Tách biên độ)</option>
                <option value="laplacian">Laplacian (Trích xuất chi tiết/cạnh)</option>
                <option value="sharpen">Sharpening (Làm sắc nét ảnh)</option>
            </select>
        </div>

        <!-- Tải ảnh / Chụp / Dán -->
        <div class="border-2 border-dashed border-slate-300 rounded-2xl p-6 text-center transition-all" id="dropZone">
            <input type="file" id="fileInput" accept="image/*" class="hidden">
            
            <div id="previewContainer" class="hidden mb-4">
                <img id="imagePreview" src="" alt="Ảnh dự đoán" class="max-h-64 mx-auto rounded-lg shadow-md mb-2">
            </div>

            <div id="uploadPrompt" class="space-y-4">
                <p class="text-slate-600">Kéo thả ảnh vào đây, nhấn <kbd class="px-2 py-1 bg-slate-100 border rounded text-xs">Ctrl + V</kbd> để dán ảnh, hoặc chọn phương thức bên dưới:</p>
                
                <div class="flex justify-center gap-3 flex-wrap">
                    <button onclick="document.getElementById('fileInput').click()" class="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition">📁 Chọn File</button>
                    <button onclick="openCamera()" class="px-4 py-2 bg-emerald-600 text-white rounded-lg hover:bg-emerald-700 transition">📷 Chụp Ảnh</button>
                </div>
            </div>
        </div>

        <!-- Modal Webcam -->
        <div id="cameraModal" class="fixed inset-0 bg-black/60 hidden items-center justify-center z-50">
            <div class="bg-white p-6 rounded-2xl max-w-md w-full text-center">
                <video id="webcam" autoplay playsinline class="w-full h-64 bg-black rounded-lg mb-4 object-cover"></video>
                <div class="flex justify-center gap-3">
                    <button onclick="captureCamera()" class="px-5 py-2 bg-emerald-600 text-white rounded-lg">Chụp</button>
                    <button onclick="closeCamera()" class="px-5 py-2 bg-slate-300 rounded-lg">Hủy</button>
                </div>
            </div>
        </div>

        <!-- Nút gửi dự đoán -->
        <button id="predictBtn" onclick="submitPrediction()" disabled class="w-full mt-6 py-3 bg-slate-300 text-slate-500 font-semibold rounded-xl cursor-not-allowed transition">
            Dự Đoán
        </button>

        <!-- Kết quả dự đoán -->
        <div id="resultBox" class="mt-6 p-4 bg-slate-100 rounded-xl hidden">
            <p class="text-slate-600 text-sm">Kết quả phân loại / dự đoán:</p>
            <p id="resultLabel" class="text-xl font-bold text-slate-800 mt-1"></p>
        </div>
    </div>

    <script>
        let currentFile = null;
        let videoStream = null;

        const dropZone = document.getElementById('dropZone');
        const fileInput = document.getElementById('fileInput');
        const imagePreview = document.getElementById('imagePreview');
        const previewContainer = document.getElementById('previewContainer');
        const predictBtn = document.getElementById('predictBtn');

        fileInput.addEventListener('change', (e) => {
            if (e.target.files.length > 0) handleFileSelect(e.target.files[0]);
        });

        document.addEventListener('paste', (e) => {
            const items = e.clipboardData.items;
            for (let item of items) {
                if (item.type.indexOf('image') !== -1) {
                    handleFileSelect(item.getAsFile());
                    break;
                }
            }
        });

        dropZone.addEventListener('dragover', (e) => { e.preventDefault(); dropZone.classList.add('drop-zone--over'); });
        dropZone.addEventListener('dragleave', () => dropZone.classList.remove('drop-zone--over'));
        dropZone.addEventListener('drop', (e) => {
            e.preventDefault();
            dropZone.classList.remove('drop-zone--over');
            if (e.dataTransfer.files.length > 0) handleFileSelect(e.dataTransfer.files[0]);
        });

        function handleFileSelect(file) {
            currentFile = file;
            const reader = new FileReader();
            reader.onload = (e) => {
                imagePreview.src = e.target.result;
                previewContainer.classList.remove('hidden');
                predictBtn.disabled = false;
                predictBtn.classList.remove('bg-slate-300', 'text-slate-500', 'cursor-not-allowed');
                predictBtn.classList.add('bg-blue-600', 'text-white', 'hover:bg-blue-700');
            };
            reader.readAsDataURL(file);
        }

        async function openCamera() {
            const modal = document.getElementById('cameraModal');
            const video = document.getElementById('webcam');
            try {
                videoStream = await navigator.mediaDevices.getUserMedia({ video: true });
                video.srcObject = videoStream;
                modal.classList.remove('hidden');
                modal.classList.add('flex');
            } catch (err) {
                alert("Không thể truy cập Camera.");
            }
        }

        function closeCamera() {
            if (videoStream) videoStream.getTracks().forEach(track => track.stop());
            document.getElementById('cameraModal').classList.add('hidden');
        }

        function captureCamera() {
            const video = document.getElementById('webcam');
            const canvas = document.createElement('canvas');
            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;
            canvas.getContext('2d').drawImage(video, 0, 0);
            
            canvas.toBlob((blob) => {
                handleFileSelect(new File([blob], "camera.jpg", { type: "image/jpeg" }));
                closeCamera();
            }, 'image/jpeg');
        }

        async function submitPrediction() {
            if (!currentFile) return;

            const selectedKernel = document.getElementById('kernelSelect').value;
            const formData = new FormData();
            formData.append('file', currentFile);
            formData.append('kernel', selectedKernel);

            predictBtn.innerText = "Đang xử lý...";
            predictBtn.disabled = true;

            try {
                const response = await fetch('/api/predict', { method: 'POST', body: formData });
                const data = await response.json();
                if (data.status === 'success') {
                    document.getElementById('resultBox').classList.remove('hidden');
                    document.getElementById('resultLabel').innerText = data.label;
                }
            } catch (err) {
                alert("Đã xảy ra lỗi.");
            } finally {
                predictBtn.innerText = "Dự Đoán";
                predictBtn.disabled = false;
            }
        }
    </script>
</body>
</html>
"""

# 2. Trang chủ hiển thị giao diện
@app.get("/", response_class=HTMLResponse)
async def read_root():
    return HTML_CONTENT

# 3. Hàm áp dụng bộ lọc Kernel
def apply_kernel(image_bytes: bytes, kernel_type: str) -> Image.Image:
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    
    if kernel_type == "gaussian":
        return img.filter(ImageFilter.GaussianBlur(radius=2))
    elif kernel_type == "sharpen":
        return img.filter(ImageFilter.SHARPEN)
    elif kernel_type in ["sobel", "prewitt"]:
        return img.filter(ImageFilter.FIND_EDGES)
    elif kernel_type == "laplacian":
        return img.filter(ImageFilter.CONTOUR)
    
    return img

# 4. Route xử lý dự đoán
@app.post("/api/predict")
async def predict(file: UploadFile = File(...), kernel: str = Form("none")):
    contents = await file.read()
    processed_img = apply_kernel(contents, kernel)
    
    # Đoạn dự đoán mô hình (thay thế bằng model AI của bạn)
    categories = ["Iris setosa", "Iris versicolor", "Iris virginica"]
    label_result = categories[np.random.randint(0, 3)]

    return {
        "status": "success",
        "label": label_result
    }
