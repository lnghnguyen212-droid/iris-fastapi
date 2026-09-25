import io
import numpy as np
from PIL import Image, ImageFilter
from fastapi import FastAPI, File, UploadFile, Form, HTTPException

app = FastAPI()

# Định nghĩa các Kernel bằng PIL và Numpy thuần
def apply_selected_kernel(image_bytes: bytes, kernel_type: str) -> Image.Image:
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    if kernel_type == "gaussian":
        return img.filter(ImageFilter.GaussianBlur(radius=2))
    
    elif kernel_type == "sharpen":
        return img.filter(ImageFilter.SHARPEN)
    
    elif kernel_type == "laplacian":
        # Kernel Laplacian 3x3
        kernel_3x3 = ImageFilter.Kernel(
            size=(3, 3),
            kernel=[-1, -1, -1, -1, 8, -1, -1, -1, -1],
            scale=1
        )
        return img.filter(kernel_3x3)

    elif kernel_type == "sobel":
        # Kernel Sobel phát hiện cạnh
        kernel_3x3 = ImageFilter.Kernel(
            size=(3, 3),
            kernel=[-1, 0, 1, -2, 0, 2, -1, 0, 1],
            scale=1
        )
        return img.filter(kernel_3x3)

    elif kernel_type == "prewitt":
        kernel_3x3 = ImageFilter.Kernel(
            size=(3, 3),
            kernel=[-1, 0, 1, -1, 0, 1, -1, 0, 1],
            scale=1
        )
        return img.filter(kernel_3x3)

    elif kernel_type == "ridge":
        kernel_3x3 = ImageFilter.Kernel(
            size=(3, 3),
            kernel=[-1, -1, -1, -1, 8, -1, -1, -1, -1],
            scale=1
        )
        return img.filter(kernel_3x3)

    elif kernel_type == "edge_enhance":
        return img.filter(ImageFilter.EDGE_ENHANCE)

    return img


@app.post("/api/predict")
async def predict_image(
    file: UploadFile = File(...),
    kernel: str = Form("none")
):
    contents = await file.read()
    
    # Xử lý ảnh qua Kernel đã chọn
    processed_img = apply_selected_kernel(contents, kernel)
    
    # Mock kết quả dự đoán (Thay thế bằng model AI của bạn)
    categories = ["Iris setosa", "Iris versicolor", "Iris virginica"]
    predicted_label = categories[np.random.randint(0, len(categories))]

    return {
        "status": "success",
        "label": predicted_label
    }
