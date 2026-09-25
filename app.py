import io
import numpy as np
from PIL import Image, ImageFilter
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import HTMLResponse

app = FastAPI()

# 1. Thêm Route hiển thị giao diện trang chủ (Fix lỗi Not Found)
@app.get("/", response_class=HTMLResponse)
async def read_root():
    with open("index.html", "r", encoding="utf-8") as f:
        return f.read()

# 2. Hàm lọc ảnh đơn giản
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

# 3. Route xử lý dự đoán
@app.post("/api/predict")
async def predict(file: UploadFile = File(...), kernel: str = Form("none")):
    contents = await file.read()
    processed_img = apply_kernel(contents, kernel)
    
    # Dự đoán (thay bằng model của bạn)
    categories = ["Iris setosa", "Iris versicolor", "Iris virginica"]
    label_result = categories[np.random.randint(0, 3)]

    return {
        "status": "success",
        "label": label_result
    }
