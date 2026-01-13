# schemas.py
INITIAL_GIFTS = [
    ("Американо", 100),
    ("Круассан", 150),
    ("Двойной эспрессо", 80),
    ("Капучино", 120)
]

INITIAL_NOTIFICATIONS = [
    ("novelty", "Лавандовый латте", "Нежный вкус прованской лаванды в вашей чашке.", 30),
    ("promotion", "Приведи друга", "Приведи друга — получи 50 бонусов!", 14),
    ("announcement", "Мастер-класс", "Завтра — бесплатный мастер-класс по латте-арту!", 1)
]

def init_database(conn):
    """Инициализация базы данных"""
    cursor = conn.cursor()
    
    # Таблица клиентов
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS clients (
            id SERIAL PRIMARY KEY,
            telegram_id BIGINT UNIQUE NOT NULL,
            card_number VARCHAR(20) UNIQUE NOT NULL,
            first_name VARCHAR(50) NOT NULL,
            last_name VARCHAR(50) NOT NULL,
            email VARCHAR(100),
            birth_date DATE,
            gender VARCHAR(10),
            points INTEGER DEFAULT 0,
            total_earned_points INTEGER DEFAULT 0,
            created_at TIMESTAMP DEFAULT NOW()
        )
    ''')
    
    # Таблица сотрудников
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS staff (
            id SERIAL PRIMARY KEY,
            telegram_id BIGINT UNIQUE NOT NULL,
            name VARCHAR(100) NOT NULL,
            role VARCHAR(20) DEFAULT 'staff' CHECK (role IN ('staff', 'admin')),
            created_at TIMESTAMP DEFAULT NOW()
        )
    ''')
    
    # Таблица подарков
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS gifts (
            id SERIAL PRIMARY KEY,
            name VARCHAR(100) NOT NULL,
            points_cost INTEGER NOT NULL,
            is_active BOOLEAN DEFAULT true,
            created_at TIMESTAMP DEFAULT NOW()
        )
    ''')
    
    # Таблица уведомлений
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS notifications (
            id SERIAL PRIMARY KEY,
            type VARCHAR(20) NOT NULL CHECK (type IN ('promotion', 'novelty', 'announcement')),
            title VARCHAR(100) NOT NULL,
            description TEXT NOT NULL,
            image_url VARCHAR(255),
            expires_at TIMESTAMP NOT NULL,
            created_at TIMESTAMP DEFAULT NOW()
        )
    ''')
    
    # Таблица транзакций
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS transactions (
            id SERIAL PRIMARY KEY,
            client_id INTEGER NOT NULL REFERENCES clients(id) ON DELETE CASCADE,
            staff_id INTEGER REFERENCES staff(id) ON DELETE SET NULL,
            type VARCHAR(20) NOT NULL CHECK (type IN ('purchase', 'social', 'gift', 'manual', 'birthday')),
            points_change INTEGER NOT NULL,
            description TEXT,
            created_at TIMESTAMP DEFAULT NOW()
        )
    ''')

    # Таблица персональных уведомлений
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS user_notifications (
            id SERIAL PRIMARY KEY,
            telegram_id BIGINT NOT NULL,
            title VARCHAR(100) NOT NULL,
            message TEXT NOT NULL,
            is_read BOOLEAN DEFAULT false,
            created_at TIMESTAMP DEFAULT NOW()
        )
    ''')
    
    # Добавление демо-данных
    cursor.execute("SELECT COUNT(*) as cnt FROM gifts")
    if cursor.fetchone()['cnt'] == 0:
        cursor.executemany(
            "INSERT INTO gifts (name, points_cost) VALUES (%s, %s)",
            INITIAL_GIFTS
        )
    
    cursor.execute("SELECT COUNT(*) as cnt FROM notifications")
    if cursor.fetchone()['cnt'] == 0:
        from datetime import datetime, timedelta
        now = datetime.utcnow()
        notifications = [
            (nt[0], nt[1], nt[2], now + timedelta(days=nt[3]))
            for nt in INITIAL_NOTIFICATIONS
        ]
        cursor.executemany(
            "INSERT INTO notifications (type, title, description, expires_at) VALUES (%s, %s, %s, %s)",
            notifications
        )
    
    conn.commit()