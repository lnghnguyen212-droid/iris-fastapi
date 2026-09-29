import sqlite3

def register_user(username, password):
    """Đăng ký tài khoản mới"""
    try:
        conn = sqlite3.connect('users.db')
        cursor = conn.cursor()
        cursor.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, password))
        conn.commit()
        conn.close()
        return True, "Đăng ký thành công!"
    except sqlite3.IntegrityError:
        return False, "Tên đăng nhập đã tồn tại!"
    except Exception as e:
        return False, f"Lỗi: {str(e)}"

def login_user(username, password):
    """Xác thực đăng nhập"""
    conn = sqlite3.connect('users.db')
    cursor = conn.cursor()
    cursor.execute("INSERT/SELECT check...", )
    cursor.execute("SELECT * FROM users WHERE username = ? AND password = ?", (username, password))
    user = cursor.fetchone()
    conn.close()
    
    if user:
        return True, "Đăng nhập thành công!"
    else:
        return False, "Sai tài khoản hoặc mật khẩu!"
