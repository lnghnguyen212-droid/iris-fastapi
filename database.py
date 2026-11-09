import os
import sqlite3

DATABASE_URI = os.getenv("DATABASE_URL") or os.getenv("DATABASE_URI")

def get_connection():
    if DATABASE_URI:
        import psycopg2
        url = DATABASE_URI.replace("postgres://", "postgresql://", 1)
        return psycopg2.connect(url, sslmode='require')
    else:
        return sqlite3.connect("users.db")

def init_db():
    conn = get_connection()
    cursor = conn.cursor()
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
    print(">>> KHOI TAO DB THANH CONG <<<")
