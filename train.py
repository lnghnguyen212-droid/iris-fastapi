
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import joblib

# 1. Đọc dữ liệu Iris
iris = load_iris()
X = iris.data
y = iris.target

# 2. Chia dữ liệu: 80% train, 20% test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 3. Huấn luyện 4 mô hình SVM
kernels = ["linear", "rbf", "poly", "sigmoid"]
models = {}

for kernel in kernels:
    model = SVC(
        kernel=kernel,
        probability=True,
        random_state=42
    )

    model.fit(X_train, y_train)
    models[kernel] = model

    # 4. Đánh giá
    y_pred = model.predict(X_test)

    print("\nKernel:", kernel)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))

# 5. Lưu mô hình
joblib.dump(models, "svm_multi_kernels.pkl")

# Lưu riêng mô hình Linear
linear_model = SVC(kernel="linear")
linear_model.fit(X_train, y_train)
joblib.dump(linear_model, "svm_model.pkl")

print("\nĐã lưu 2 file mô hình!")
