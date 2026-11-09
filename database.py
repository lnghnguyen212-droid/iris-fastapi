import os
import sqlite3

DATABASE_URI = os.getenv("DATABASE_URL") or os.getenv("DATABASE_URI")


def get_connection():
    if DATABASE_URI:
        import psycopg2
        url = DATABASE_URI.replace("postgres://", "postgresql://", 1)
        return psycopg2.connect(url, sslmode="require")
    else:
        return sqlite3.connect("users.db")


def init_db():
    # PostgreSQL dùng SERIAL, SQLite dùng AUTOINCREMENT
    id_col = "SERIAL PRIMARY KEY" if DATABASE_URI else "INTEGER PRIMARY KEY AUTOINCREMENT"

    conn = get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute(f"""
            CREATE TABLE IF NOT EXISTS users (
                id {id_col},
                username VARCHAR(255) UNIQUE NOT NULL,
                password VARCHAR(255) NOT NULL
            );
        """)
        conn.commit()
        cursor.close()
        print(">>> KHOI TAO DB THANH CONG <<<")
    finally:
        conn.close()
