from datetime import datetime, timedelta

def init_database(conn):
    cursor = conn.cursor()
    
    # 1. Клиенты (добавлено поле phone)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS clients (
            id SERIAL PRIMARY KEY,
            telegram_id BIGINT UNIQUE NOT NULL,
            card_number VARCHAR(20) UNIQUE NOT NULL,
            first_name VARCHAR(50) NOT NULL,
            last_name VARCHAR(50) NOT NULL,
            email VARCHAR(100),
            phone VARCHAR(20),
            birth_date DATE,
            gender VARCHAR(10),
            points INTEGER DEFAULT 0,
            total_earned_points INTEGER DEFAULT 0,
            created_at TIMESTAMP DEFAULT NOW()
        )
    ''')
    
    # 2. Сотрудники
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS staff (
            id SERIAL PRIMARY KEY,
            telegram_id BIGINT UNIQUE NOT NULL,
            name VARCHAR(100) NOT NULL,
            role VARCHAR(20) DEFAULT 'staff' CHECK (role IN ('staff', 'admin')),
            created_at TIMESTAMP DEFAULT NOW()
        )
    ''')
    
    # 3. Подарки (добавлен image_url)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS gifts (
            id SERIAL PRIMARY KEY,
            name VARCHAR(100) NOT NULL,
            points_cost INTEGER NOT NULL,
            image_url VARCHAR(255),
            is_active BOOLEAN DEFAULT true,
            created_at TIMESTAMP DEFAULT NOW()
        )
    ''')
    
    # 4. Уведомления
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
    
    # 5. Транзакции (client_id теперь NULLABLE, добавлены target поля для админки)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS transactions (
            id SERIAL PRIMARY KEY,
            client_id INTEGER REFERENCES clients(id) ON DELETE CASCADE,
            staff_id INTEGER REFERENCES staff(id) ON DELETE SET NULL,
            type VARCHAR(50) NOT NULL,
            points_change INTEGER NOT NULL DEFAULT 0,
            description TEXT,
            target_type VARCHAR(50),
            target_id INTEGER,
            created_at TIMESTAMP DEFAULT NOW()
        )
    ''')

    # 6. Персональные уведомления
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
    
    conn.commit()
    
    # Заполнение демо-данными (только если пусто)
    cursor.execute("SELECT COUNT(*) FROM gifts")
    if cursor.fetchone()['count'] == 0:
        gifts = [("Американо", 100), ("Круассан", 150), ("Капучино", 120)]
        cursor.executemany("INSERT INTO gifts (name, points_cost) VALUES (%s, %s)", gifts)
    
    conn.commit()