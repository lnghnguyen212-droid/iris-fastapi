import os
import sqlite3

# Hỗ trợ đọc cả 2 tên biến để tránh việc cấu hình nhầm trên Render
DATABASE_URI = os.getenv("DATABASE_URL") or os.getenv("DATABASE_URI")

def get_connection():
    if DATABASE_URI:
        import psycopg2
        # Render cung cấp url dạng postgres://, cần đổi thành postgresql:// cho psycopg2
        url = DATABASE_URI.replace("postgres://", "postgresql://", 1)
        
        # Thêm sslmode='require' để bắt buộc kết nối an toàn với PostgreSQL trên Render
        return psycopg2.connect(url, sslmode='require')
    else:
        return sqlite3.connect("users.db")

def init_db():
    conn = get_connection()
    cursor = conn.cursor()
    
    # Tự động tạo bảng users nếu chưa tồn tại (chạy cho cả PostgreSQL trên Render lẫn SQLite local)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id SERIAL PRIMARY KEY,
            username VARCHAR(255) UNIQUE NOT NULL,
            password VARCHAR(255) NOT NULL
        );
    """)
    
    conn.commit()
    cursor.close()
    conn.close()
    print(">>> ĐÃ KHỞI TẠO BẢNG USERS THÀNH CÔNG! <<<")
