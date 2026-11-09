from database import get_connection, DATABASE_URI

# Dùng chung cờ với database.py để không bị lệch biến môi trường
IS_POSTGRES = bool(DATABASE_URI)
PH = "%s" if IS_POSTGRES else "?"  # PostgreSQL dùng %s, SQLite dùng ?


def register_user(username, password):
    """Đăng ký tài khoản mới"""
    conn = None
    try:
        conn = get_connection()
        cursor = conn.cursor()
        query = f"INSERT INTO users (username, password) VALUES ({PH}, {PH})"
        cursor.execute(query, (username, password))
        conn.commit()
        return True, "Đăng ký thành công!"

    except Exception as e:
        if conn:
            conn.rollback()
        err_msg = str(e)
        if "UNIQUE" in err_msg.upper() or "DUPLICATE" in err_msg.upper():
            return False, "Tên đăng nhập đã tồn tại!"
        return False, f"Lỗi: {err_msg}"

    finally:
        if conn:
            conn.close()


def login_user(username, password):
    """Xác thực đăng nhập"""
    conn = None
    try:
        conn = get_connection()
        cursor = conn.cursor()
        query = f"SELECT * FROM users WHERE username = {PH} AND password = {PH}"
        cursor.execute(query, (username, password))
        user = cursor.fetchone()

        if user:
            return True, "Đăng nhập thành công!"
        return False, "Sai tài khoản hoặc mật khẩu!"

    except Exception as e:
        return False, f"Lỗi: {str(e)}"

    finally:
        if conn:
            conn.close()
