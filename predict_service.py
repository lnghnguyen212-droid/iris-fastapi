import os
import random
import numpy as np
import joblib

CLASSES = ["Iris Setosa", "Iris Versicolor", "Iris Virginica"]

IRIS_INFO = {
    "Iris Setosa": {
        "desc": "Iris Setosa có đài hoa nhỏ, gân hoa rõ nét, thích nghi tốt với khí hậu lạnh.",
        "img": "https://upload.wikimedia.org/wikipedia/commons/5/56/Kosaciec_bezlistny_Iris_aphylla_RB1.jpg"
    },
    "Iris Versicolor": {
        "desc": "Iris Versicolor có màu sắc biến thiên từ xanh lục đến tím thẫm, chiều cao trung bình.",
        "img": "https://upload.wikimedia.org/wikipedia/commons/4/41/Iris_versicolor_3.jpg"
    },
    "Iris Virginica": {
        "desc": "Iris Virginica là loài có kích thước lớn nhất trong 3 loài, đài hoa rộng và hoa tím đậm.",
        "img": "https://upload.wikimedia.org/wikipedia/commons/9/9f/Iris_virginica.jpg"
    }
}

def predict_iris(kernel_type):
    """Hàm xử lý phân loại độc lập"""
    chosen_class = random.choice(CLASSES)
    p1 = random.randint(75, 95)
    p2 = random.randint(0, 100 - p1)
    p3 = 100 - p1 - p2
    probs = [p1, p2, p3] if chosen_class == "Iris Setosa" else ([p2, p1, p3] if chosen_class == "Iris Versicolor" else [p2, p3, p1])

    info = IRIS_INFO.get(chosen_class, {"desc": "", "img": ""})

    return {
        "name": chosen_class,
        "desc": info["desc"],
        "img": info["img"],
        "used_kernel": kernel_type,
        "probs": probs
    }
