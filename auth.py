import os
from database import get_connection

IS_POSTGRES = bool(os.getenv("DATABASE_URL"))

def register_user(username, password):
    """Đăng ký tài khoản mới"""
    try:
        conn = get_connection()
        cursor = conn.cursor()

        # PostgreSQL dùng %s, SQLite dùng ?
        placeholder = "%s" if IS_POSTGRES else "?"
        query = f"INSERT INTO users (username, password) VALUES ({placeholder}, {placeholder})"
        
        cursor.execute(query, (username, password))
        conn.commit()
        conn.close()

        return True, "Đăng ký thành công!"

    except Exception as e:
        err_msg = str(e)
        if "UNIQUE" in err_msg.upper() or "DUPLICATE" in err_msg.upper():
            return False, "Tên đăng nhập đã tồn tại!"
        return False, f"Lỗi: {err_msg}"


def login_user(username, password):
    """Xác thực đăng nhập"""
    try:
        conn = get_connection()
        cursor = conn.cursor()

        placeholder = "%s" if IS_POSTGRES else "?"
        query = f"SELECT * FROM users WHERE username = {placeholder} AND password = {placeholder}"
        
        cursor.execute(query, (username, password))
        user = cursor.fetchone()
        conn.close()

        if user:
            return True, "Đăng nhập thành công!"
        else:
            return False, "Sai tài khoản hoặc mật khẩu!"

    except Exception as e:
        return False, f"Lỗi: {str(e)}"
