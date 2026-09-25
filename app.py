import io
import numpy as np
from PIL import Image, ImageFilter
from fastapi import FastAPI, File, UploadFile, Form

app = FastAPI()

# Hàm áp dụng Kernel lọc ảnh đơn giản dùng Pillow
def apply_kernel(image_bytes: bytes, kernel_type: str) -> Image.Image:
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    
    if kernel_type == "gaussian":
        return img.filter(ImageFilter.GaussianBlur(radius=2))
    elif kernel_type == "sharpen":
        return img.filter(ImageFilter.SHARPEN)
    elif kernel_type == "sobel" or kernel_type == "prewitt":
        # Bộ lọc phát hiện cạnh cơ bản
        return img.filter(ImageFilter.FIND_EDGES)
    elif kernel_type == "laplacian":
        return img.filter(ImageFilter.CONTOUR)
    
    return img

@app.post("/api/predict")
async def predict(file: UploadFile = File(...), kernel: str = Form("none")):
    contents = await file.read()
    
    # Xử lý ảnh qua bộ lọc
    processed_img = apply_kernel(contents, kernel)
    
    # --- ĐOẠN DỰ ĐOÁN MÔ HÌNH CỦA BẠN ---
    # Thay thế phần này bằng code dự đoán từ mô hình Iris / AI của bạn
    categories = ["Iris setosa", "Iris versicolor", "Iris virginica"]
    label_result = categories[np.random.randint(0, 3)]
    # ------------------------------------

    return {
        "status": "success",
        "label": label_result
    }
