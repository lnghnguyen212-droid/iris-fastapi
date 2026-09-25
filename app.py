import io
import cv2
import numpy as np
from PIL import Image
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

app = FastAPI()

# Map các kernel xử lý ảnh phổ biến
KERNELS = {
    "gaussian": lambda img: cv2.GaussianBlur(img, (5, 5), 0),
    "sobel": lambda img: cv2.Sobel(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.CV_64F, 1, 1, ksize=3),
    "prewitt": lambda img: cv2.filter2D(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), -1, np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])),
    "laplacian": lambda img: cv2.Laplacian(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.CV_64F),
    "sharpen": lambda img: cv2.filter2D(img, -1, np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])),
    "ridge": lambda img: cv2.filter2D(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), -1, np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])),
    "canny": lambda img: cv2.Canny(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 100, 200),
    "gabor": lambda img: cv2.filter2D(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), -1, cv2.getGaborKernel((21, 21), 5.0, np.pi/4, 10.0, 0.5, 0, ktype=cv2.CV_32F))
}

def apply_selected_kernel(image_bytes: bytes, kernel_type: str):
    np_img = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
    
    if img is None:
        raise HTTPException(status_code=400, detail="Ảnh không hợp lệ")

    if kernel_type in KERNELS:
        processed_img = KERNELS[kernel_type](img)
        if len(processed_img.shape) == 2 or processed_img.dtype != np.uint8:
            processed_img = cv2.normalize(processed_img, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        return processed_img
    return img

@app.post("/api/predict")
async def predict_image(
    file: UploadFile = File(...),
    kernel: str = Form("none")
):
    contents = await file.read()
    
    # Xử lý qua Kernel được chọn (nếu có)
    processed_img = apply_selected_kernel(contents, kernel)
    
    # Mock kết quả dự đoán (Thay thế đoạn này bằng mô hình AI/ML của bạn)
    # Ví dụ trả về nhãn nhận diện được mà KHÔNG kèm % độ chính xác
    categories = ["Nhãn A (Mẫu chuẩn)", "Nhãn B (Bình thường)", "Nhãn C (Cần kiểm tra)"]
    predicted_label = categories[np.random.randint(0, len(categories))]

    # Encode ảnh đã xử lý để trả về giao diện xem trước nếu cần
    _, buffer = cv2.imencode('.jpg', processed_img)
    
    return {
        "status": "success",
        "label": predicted_label
    }
