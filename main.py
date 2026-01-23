import os
import hmac
import hashlib
import json
import time
import secrets
import string
import logging
import asyncio
import aiohttp
import html
from pathlib import Path
from contextlib import contextmanager, asynccontextmanager
from urllib.parse import parse_qsl

from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter
from slowapi.util import get_remote_address
from dotenv import load_dotenv
from datetime import datetime, timezone, timedelta
from typing import Optional
from pydantic import BaseModel, HttpUrl 

# === ИНИЦИАЛИЗАЦИЯ ===
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv(Path(__file__).parent.parent / ".env")

BOT_TOKEN = os.getenv("BOT_TOKEN")
if not BOT_TOKEN:
    raise ValueError("⚠️ BOT_TOKEN не найден в .env")

# === УПРАВЛЕНИЕ СЕССИЯМИ TELEGRAM ===
class TelegramBot:
    session: aiohttp.ClientSession = None

    @classmethod
    async def get_session(cls) -> aiohttp.ClientSession:
        if cls.session is None or cls.session.closed:
            cls.session = aiohttp.ClientSession()
        return cls.session

    @classmethod
    async def close(cls):
        if cls.session and not cls.session.closed:
            await cls.session.close()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup logic
    from database import get_db_connection
    from schemas import init_database
    conn = None
    try:
        conn = get_db_connection()
        init_database(conn)
        logger.info("✅ База данных успешно инициализирована")
    except Exception as e:
        logger.error(f"❌ Ошибка инициализации БД: {e}")
    finally:
        if conn: conn.close()
    
    yield
    # Shutdown logic
    await TelegramBot.close()

# === FastAPI APP ===
app = FastAPI(
    title="Система лояльности DwnTwn",
    description="Production-ready API для сети кофеен",
    version="1.0.0",
    lifespan=lifespan
)

# === CORS ===
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://dwntwn-loyalty-frontend-io.vercel.app",
        "https://web.telegram.org",
        "https://t.me"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# === Rate Limiter ===
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter

# === ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ===

def normalize_phone(phone: str | None) -> str | None:
    if not phone: return None
    digits = ''.join(filter(str.isdigit, phone))
    if len(digits) == 11 and digits.startswith(('7', '8')):
        return f"+7{digits[1:]}"
    if len(digits) == 10:
        return f"+7{digits}"
    raise ValueError("Некорректный формат номера телефона")

def generate_card_number(conn) -> str:
    prefix = "DTLC"
    cursor = conn.cursor()
    for _ in range(10):
        suffix = ''.join(secrets.choice(string.digits) for _ in range(6))
        card_number = f"{prefix}-{suffix}"
        cursor.execute("SELECT 1 FROM clients WHERE card_number = %s", (card_number,))
        if not cursor.fetchone():
            return card_number
    raise RuntimeError("Не удалось сгенерировать уникальный номер карты")

def get_level(points: int) -> str:
    if points >= 1000: return "PLATINA"
    if points >= 500: return "GOLD"
    if points >= 300: return "SILVER"
    if points >= 100: return "BRONZE"
    return "IRON"

@contextmanager
def get_db():
    from database import get_db_connection
    conn = get_db_connection()
    try:
        yield conn
    finally:
        conn.close()

# === ТЕЛЕГРАМ ФУНКЦИИ ===

def escape_html(text: str) -> str:
    return html.escape(text, quote=False)

async def send_telegram_message(telegram_id: int, text: str):
    """Единая функция отправки сообщений через общую сессию"""
    session = await TelegramBot.get_session()
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": telegram_id,
        "text": text,
        "parse_mode": "HTML"
    }
    try:
        async with session.post(url, json=payload) as response:
            if response.status == 429:
                retry_after = int(response.headers.get("Retry-After", 1))
                await asyncio.sleep(retry_after)
                return await send_telegram_message(telegram_id, text)
            if response.status != 200:
                error_data = await response.json()
                if error_data.get("error_code") != 403:
                    logger.warning(f"TG Error {telegram_id}: {error_data}")
    except Exception as e:
        logger.error(f"Сетевая ошибка TG {telegram_id}: {e}")

async def broadcast_new_gift(gift_name: str, points_cost: int):
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT telegram_id FROM clients")
        users = cursor.fetchall()

    if not users: return

    safe_name = escape_html(gift_name)
    text = (
        f"🎁 <b>У нас новый подарок!</b>\n\n"
        f"Теперь вы можете обменять баллы на: <b>{safe_name}</b>\n"
        f"Стоимость: <b>{points_cost}</b> баллов.\n\n"
        f"Заглядывайте в приложение! ☕️"
    )

    for user in users:
        await send_telegram_message(user['telegram_id'], text)
        await asyncio.sleep(0.04) # Лимит 30 сообщений в секунду

# === ВАЛИДАЦИЯ И AUTH ===

def validate_telegram_init_data(init_data: str) -> dict:
    if not init_data:
        raise HTTPException(status_code=401, detail="Missing initData")
    try:
        parsed_data = dict(parse_qsl(init_data))
        if "hash" not in parsed_data:
            raise HTTPException(status_code=401, detail="Missing hash")
        
        received_hash = parsed_data.pop("hash")
        auth_date = int(parsed_data.get("auth_date", 0))
        
        if time.time() - auth_date > 86400:
            raise HTTPException(status_code=401, detail="Telegram data expired")

        data_check_string = "\n".join(f"{k}={v}" for k, v in sorted(parsed_data.items()))
        secret_key = hmac.new(b"WebAppData", BOT_TOKEN.encode(), hashlib.sha256).digest()
        expected_hash = hmac.new(secret_key, data_check_string.encode(), hashlib.sha256).hexdigest()

        if not hmac.compare_digest(expected_hash, received_hash):
            raise HTTPException(status_code=401, detail="Data integrity error")

        if "user" in parsed_data:
            parsed_data["user"] = json.loads(parsed_data["user"])
        return parsed_data
    except Exception as e:
        if isinstance(e, HTTPException): raise e
        raise HTTPException(status_code=400, detail="Invalid initData format")

def extract_telegram_id(init_data: str) -> int:
    parsed = validate_telegram_init_data(init_data)
    user = parsed.get("user")
    if not user or "id" not in user:
        raise HTTPException(status_code=401, detail="Invalid user data")
    return int(user["id"])

# === МОДЕЛИ ===

class AuthUser(BaseModel):
    telegram_id: int
    role: str

class ClientRegister(BaseModel):
    telegram_id: int
    first_name: str
    last_name: str
    phone: str | None = None
    email: str | None = None
    birth_date: str | None = None
    gender: str | None = None

# === ЗАВИСИМОСТИ ===

async def get_current_user(request: Request) -> AuthUser:
    """
    Важное исправление: Мы читаем JSON один раз и сохраняем 
    его в request.state, чтобы эндпоинты могли прочитать его снова.
    """
    try:
        body = await request.json()
        request.state.body = body # Кешируем тело для эндпоинта
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON")

    init_data = body.get("initData")
    if not init_data:
        raise HTTPException(status_code=401, detail="initData is required")

    telegram_id = extract_telegram_id(init_data)

    with get_db() as conn:
        cursor = conn.cursor()
        # Сначала проверяем персонал
        cursor.execute("SELECT role FROM staff WHERE telegram_id = %s", (telegram_id,))
        staff = cursor.fetchone()
        if staff:
            return AuthUser(telegram_id=telegram_id, role=staff["role"])

        # Затем клиентов
        cursor.execute("SELECT 1 FROM clients WHERE telegram_id = %s", (telegram_id,))
        if cursor.fetchone():
            return AuthUser(telegram_id=telegram_id, role="client")

        raise HTTPException(status_code=403, detail="User not registered")

async def require_staff(user: AuthUser = Depends(get_current_user)) -> AuthUser:
    if user.role not in ("staff", "admin"):
        raise HTTPException(status_code=403, detail="Staff access required")
    return user

async def require_admin(user: AuthUser = Depends(get_current_user)) -> AuthUser:
    if user.role != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")
    return user

# === ЭНДПОИНТЫ ===

@app.post("/api/client/check-registered")
@limiter.limit("20/minute")
async def check_registered(request: Request):
    """
    Проверка регистрации. 
    Важно: эндпоинт открыт, но защищен лимитами.
    """
    try:
        body = await request.json()
        telegram_id = body.get("telegram_id")
        if not telegram_id:
            raise HTTPException(status_code=400, detail="telegram_id required")
        
        with get_db() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT 1 FROM clients WHERE telegram_id = %s", (telegram_id,))
            exists = cursor.fetchone() is not None
            return {"registered": exists}
    except Exception as e:
        logger.error(f"Check registered error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.post("/api/client/register")
@limiter.limit("5/minute")
async def register_client(request: Request):
    """
    Регистрация нового клиента. 
    Использует валидацию данных Telegram внутри функции.
    """
    try:
        body = await request.json()
        init_data = body.get("initData")
        if not init_data:
            raise HTTPException(status_code=400, detail="initData is required")

        # 1. Валидация и извлечение ID
        telegram_id = extract_telegram_id(init_data)

        # 2. Нормализация телефона
        raw_phone = body.get("phone")
        if not raw_phone:
             raise HTTPException(status_code=400, detail="Номер телефона обязателен")
        
        normalized_phone = normalize_phone(raw_phone)

        # 3. Валидация через Pydantic
        client_data = ClientRegister(
            telegram_id=telegram_id,
            first_name=body.get("first_name", "").strip(),
            last_name=body.get("last_name", "").strip(),
            phone=normalized_phone,
            email=body.get("email", "").strip() or None,
            birth_date=body.get("birth_date"),
            gender=body.get("gender")
        )

        with get_db() as conn:
            with conn.cursor() as cursor:
                # Проверка дубликата
                cursor.execute("SELECT 1 FROM clients WHERE telegram_id = %s", (telegram_id,))
                if cursor.fetchone():
                    raise HTTPException(status_code=400, detail="Вы уже зарегистрированы")

                # Регистрация
                card_number = generate_card_number(conn)
                cursor.execute("""
                    INSERT INTO clients (
                        telegram_id, card_number, first_name, last_name, email, phone, birth_date, gender, points, total_earned_points
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, 0, 0)
                """, (
                    client_data.telegram_id, card_number, client_data.first_name, client_data.last_name,
                    client_data.email, client_data.phone, client_data.birth_date, client_data.gender
                ))
                conn.commit()

        # Приветственное сообщение
        welcome_text = f"🎉 <b>{escape_html(client_data.first_name)}, добро пожаловать!</b>\nВаша карта {card_number} активирована."
        asyncio.create_task(send_telegram_message(telegram_id, welcome_text))

        return {"card_number": card_number}

    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logger.error(f"Registration Error: {e}")
        if isinstance(e, HTTPException): raise e
        raise HTTPException(status_code=500, detail="Ошибка при регистрации")

@app.post("/api/client/profile")
@limiter.limit("15/minute")
async def get_profile(request: Request, user: AuthUser = Depends(get_current_user)):
    if user.role != "client":
        raise HTTPException(status_code=403, detail="Доступ только для клиентов")
    
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT card_number, first_name, last_name, points, total_earned_points, birth_date
            FROM clients WHERE telegram_id = %s
        """, (user.telegram_id,))
        client = cursor.fetchone()
        
        if not client:
            raise HTTPException(status_code=404, detail="Клиент не найден")
            
        return {
            "card_number": client["card_number"],
            "first_name": client["first_name"],
            "last_name": client["last_name"],
            "points": client["points"],
            "total_earned_points": client["total_earned_points"],
            "level": get_level(client["total_earned_points"]),
            "telegram_id": user.telegram_id,
            "birth_date": str(client["birth_date"]) if client["birth_date"] else None
        }

@app.post("/api/client/transactions")
@limiter.limit("10/minute")
async def get_client_transactions(request: Request, user: AuthUser = Depends(get_current_user)):
    if user.role != "client":
        raise HTTPException(status_code=403, detail="Client access required")
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT id, type, points_change, description, created_at
            FROM transactions
            WHERE client_id = (SELECT id FROM clients WHERE telegram_id = %s)
            ORDER BY created_at DESC
        """, (user.telegram_id,))
        return cursor.fetchall()

@app.post("/api/client/notifications")
@limiter.limit("10/minute")
async def get_notifications(request: Request, user: AuthUser = Depends(get_current_user)):
    with get_db() as conn:
        cursor = conn.cursor()
        now = datetime.utcnow()
        cursor.execute("""
            SELECT id, type, title, description, image_url, expires_at
            FROM notifications WHERE expires_at > %s ORDER BY expires_at DESC
        """, (now,))
        return cursor.fetchall()

@app.post("/api/client/gifts")
async def get_gifts(request: Request, user: AuthUser = Depends(get_current_user)):
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT id, name, points_cost, image_url 
            FROM gifts 
            WHERE is_active = true 
            ORDER BY points_cost ASC
        """)
        return cursor.fetchall()

@app.post("/api/client/delete-account")
@limiter.limit("2/minute")
async def delete_account(request: Request, user: AuthUser = Depends(get_current_user)):
    if user.role != "client":
        raise HTTPException(status_code=403, detail="Only clients can delete their account")
    
    with get_db() as conn:
        with conn.cursor() as cursor:
            try:
                # Находим ID клиента
                cursor.execute("SELECT id FROM clients WHERE telegram_id = %s", (user.telegram_id,))
                res = cursor.fetchone()
                if not res:
                    raise HTTPException(status_code=404, detail="Account not found")
                
                cid = res["id"]
                # Удаляем связанные данные
                cursor.execute("DELETE FROM user_notifications WHERE telegram_id = %s", (user.telegram_id,))
                cursor.execute("DELETE FROM transactions WHERE client_id = %s", (cid,))
                cursor.execute("DELETE FROM clients WHERE id = %s", (cid,))
                
                conn.commit()
                
                farewell = "🙏 Ваши данные удалены. Будем рады видеть вас снова!"
                asyncio.create_task(send_telegram_message(user.telegram_id, farewell))
                return {"status": "ok"}
                
            except Exception as e:
                conn.rollback()
                logger.error(f"Delete error: {e}")
                raise HTTPException(status_code=500, detail="Ошибка удаления")

# === СОТРУДНИКИ ===

@app.post("/api/staff/login")
@limiter.limit("5/minute")
async def staff_login(request: Request, user: AuthUser = Depends(require_staff)):
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT id, name, role FROM staff WHERE telegram_id = %s", (user.telegram_id,))
        staff = cursor.fetchone()
        if not staff:
            raise HTTPException(status_code=404, detail="Сотрудник не найден")
        return staff

@app.post("/api/staff/my-transactions")
@limiter.limit("15/minute")
async def get_staff_transactions(request: Request, user: AuthUser = Depends(require_staff)):
    with get_db() as conn:
        cursor = conn.cursor()
        
        # Получаем ID сотрудника
        cursor.execute("SELECT id FROM staff WHERE telegram_id = %s", (user.telegram_id,))
        staff_row = cursor.fetchone()
        if not staff_row:
            return []
        
        s_id = staff_row["id"] if isinstance(staff_row, dict) else staff_row[0]

        cursor.execute("""
            SELECT 
                t.id,
                COALESCE(CONCAT(c.first_name, ' ', c.last_name), 'Клиент удален') AS client_name,
                t.points_change,
                t.description,
                t.created_at
            FROM transactions t
            LEFT JOIN clients c ON t.client_id = c.id
            WHERE t.staff_id = %s
            ORDER BY t.created_at DESC
            LIMIT 100
        """, (s_id,))
        
        rows = cursor.fetchall()
        result = []
        for row in rows:
            # Безопасное форматирование даты
            dt = row["created_at"]
            dt_str = dt.strftime("%Y-%m-%d %H:%M") if isinstance(dt, datetime) else str(dt)

            result.append({
                "id": row["id"],
                "client_name": row["client_name"],
                "points_change": row["points_change"],
                "description": row["description"],
                "created_at": dt_str
            })
        return result

@app.post("/api/staff/client-by-card")
@limiter.limit("20/minute")
async def get_client_by_card(request: Request, user: AuthUser = Depends(require_staff)):
    body = await request.json()
    card_number = body.get("card_number")
    if not card_number:
        raise HTTPException(status_code=400, detail="Укажите номер карты")

    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT id, telegram_id, first_name, last_name, points, total_earned_points 
            FROM clients WHERE card_number = %s
        """, (card_number,))
        client = cursor.fetchone()
        
        if not client:
            raise HTTPException(status_code=404, detail="Клиент не найден")
            
        return {
            "id": client["id"],
            "name": f"{client['first_name']} {client['last_name']}".strip(),
            "points": client["points"],
            "level": get_level(client["total_earned_points"])
        }

@app.post("/api/staff/add-points")
@limiter.limit("20/minute")
async def add_points(request: Request, user: AuthUser = Depends(require_staff)):
    body = await request.json()
    client_id = body.get("client_id")
    try:
        purchase_amount = float(body.get("purchase_amount", 0))
    except (ValueError, TypeError):
        raise HTTPException(status_code=400, detail="Некорректная сумма покупки")

    if purchase_amount <= 0 or purchase_amount > 2500:
        raise HTTPException(status_code=400, detail="Сумма должна быть от 1 до 2500 руб.")

    with get_db() as conn:
        cursor = conn.cursor()
        
        # --- ПРОВЕРКА ЛИМИТОВ (Защита от накрутки) ---
        one_hour_ago = datetime.now(timezone.utc) - timedelta(hours=1)
        
        # Парсим сумму из описания транзакций за последний час
        cursor.execute("""
            SELECT description 
            FROM transactions 
            WHERE client_id = %s AND type = 'purchase' AND created_at > %s
        """, (client_id, one_hour_ago))
        
        past_txs = cursor.fetchall()
        total_spent_hour = 0
        for tx in past_txs:
            # Извлекаем число из строки "Покупка на X руб."
            try:
                parts = tx["description"].split()
                if "Покупка" in parts:
                    val = float(parts[2])
                    total_spent_hour += val
            except: continue

        if (total_spent_hour + purchase_amount) > 2500:
            raise HTTPException(
                status_code=403, 
                detail=f"Лимит 2500р/час. Уже потрачено: {total_spent_hour}р. Доступно: {2500 - total_spent_hour}р."
            )

        # --- НАЧИСЛЕНИЕ ---
        cursor.execute("SELECT points, total_earned_points, telegram_id FROM clients WHERE id = %s", (client_id,))
        client = cursor.fetchone()
        if not client:
            raise HTTPException(status_code=404, detail="Клиент не найден")

        level = get_level(client["total_earned_points"])
        multipliers = {"PLATINA": 0.10, "GOLD": 0.07, "SILVER": 0.05, "BRONZE": 0.03, "IRON": 0.01}
        bonus_points = max(1, int(purchase_amount * multipliers.get(level, 0.01)))
        
        new_balance = client["points"] + bonus_points
        new_total = client["total_earned_points"] + bonus_points

        try:
            cursor.execute("""
                UPDATE clients SET points = %s, total_earned_points = %s WHERE id = %s
            """, (new_balance, new_total, client_id))
            
            cursor.execute("""
                INSERT INTO transactions (client_id, staff_id, type, points_change, description)
                VALUES (%s, (SELECT id FROM staff WHERE telegram_id = %s), 'purchase', %s, %s)
            """, (client_id, user.telegram_id, bonus_points, f"Покупка на {purchase_amount} руб. ({level})"))
            
            conn.commit()
            
            # Уведомление клиенту
            msg = (f"☕️ <b>Начисление баллов!</b>\n\nСумма: {purchase_amount} руб.\n"
                   f"Начислено: +<b>{bonus_points}</b>\nБаланс: <b>{new_balance}</b>")
            asyncio.create_task(send_telegram_message(client["telegram_id"], msg))
            
            return {"status": "ok", "added": bonus_points, "balance": new_balance}
        except Exception as e:
            conn.rollback()
            logger.error(f"Points add error: {e}")
            raise HTTPException(status_code=500, detail="Ошибка БД")

@app.post("/api/staff/redeem-gift")
@limiter.limit("10/minute")
async def redeem_gift(request: Request, user: AuthUser = Depends(require_staff)):
    body = await request.json()
    client_id = body.get("client_id")
    gift_id = body.get("gift_id")

    with get_db() as conn:
        cursor = conn.cursor()
        
        # Проверка подарка
        cursor.execute("SELECT name, points_cost FROM gifts WHERE id = %s AND is_active = true", (gift_id,))
        gift = cursor.fetchone()
        if not gift:
            raise HTTPException(status_code=404, detail="Подарок не найден")

        # Проверка баланса
        cursor.execute("SELECT points, telegram_id FROM clients WHERE id = %s", (client_id,))
        client = cursor.fetchone()
        if not client or client["points"] < gift["points_cost"]:
            raise HTTPException(status_code=400, detail="Недостаточно баллов")

        new_points = client["points"] - gift["points_cost"]
        
        try:
            cursor.execute("UPDATE clients SET points = %s WHERE id = %s", (new_points, client_id))
            cursor.execute("""
                INSERT INTO transactions (client_id, staff_id, type, points_change, description)
                VALUES (%s, (SELECT id FROM staff WHERE telegram_id = %s), 'gift', %s, %s)
            """, (client_id, user.telegram_id, -gift["points_cost"], f"Подарок: {gift['name']}"))
            
            conn.commit()
            
            msg = (f"🎁 <b>Подарок выдан!</b>\n\n{gift['name']}\n"
                   f"Списано: <b>{gift['points_cost']}</b>\nОстаток: <b>{new_points}</b>")
            asyncio.create_task(send_telegram_message(client["telegram_id"], msg))
            
            return {"status": "ok", "new_points": new_points}
        except Exception as e:
            conn.rollback()
            raise HTTPException(status_code=500, detail="Ошибка списания")
        

# === АДМИНКА ===

@app.post("/api/admin/gifts")
@limiter.limit("5/minute")
async def get_all_gifts(request: Request, user: AuthUser = Depends(require_admin)):
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT id, name, points_cost, image_url, is_active FROM gifts ORDER BY points_cost")
        return cursor.fetchall()

@app.post("/api/admin/delete-gift")
@limiter.limit("5/minute")
async def delete_gift(request: Request, user: AuthUser = Depends(require_admin)):
    body = await request.json()
    gift_id = body.get("gift_id")
    
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM gifts WHERE id = %s", (gift_id,))
        gift = cursor.fetchone()
        if not gift:
            raise HTTPException(status_code=404, detail="Подарок не найден")

        cursor.execute("UPDATE gifts SET is_active = false WHERE id = %s", (gift_id,))
        
        audit_desc = f"Админ удалил подарок: «{gift['name']}»"
        cursor.execute("""
            INSERT INTO transactions (staff_id, type, description, points_change)
            VALUES ((SELECT id FROM staff WHERE telegram_id = %s), 'gift_deleted', %s, 0)
        """, (user.telegram_id, audit_desc))
        
        conn.commit()
        return {"status": "ok"}

@app.post("/api/admin/cancel-transaction")
@limiter.limit("5/minute")
async def cancel_transaction(request: Request, user: AuthUser = Depends(require_admin)):
    """
    Безопасная отмена операции с возвратом/списанием баллов.
    """
    body = await request.json()
    tx_id = body.get("transaction_id")

    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM transactions WHERE id = %s", (tx_id,))
        tx = cursor.fetchone()
        
        if not tx:
            raise HTTPException(status_code=404, detail="Транзакция не найдена")
        if "[ОТМЕНЕНА]" in tx["description"] or tx["type"] == 'transaction_cancelled':
            raise HTTPException(status_code=400, detail="Операция уже отменена")

        client_id = tx["client_id"]
        points_to_revert = -tx["points_change"]
        
        # Корректируем total_earned только если отменяем ПРИХОД баллов
        earned_change = points_to_revert if tx["points_change"] > 0 else 0
        
        try:
            # 1. Обновляем баланс
            cursor.execute("""
                UPDATE clients 
                SET points = points + %s, total_earned_points = total_earned_points + %s
                WHERE id = %s
            """, (points_to_revert, earned_change, client_id))

            # 2. Помечаем оригинал
            cursor.execute("UPDATE transactions SET description = %s WHERE id = %s", 
                           (f"[ОТМЕНЕНА] {tx['description']}", tx_id))

            # 3. Создаем запись об отмене
            audit_msg = f"ОТМЕНА: {tx['description']}"
            cursor.execute("""
                INSERT INTO transactions (staff_id, client_id, type, description, points_change)
                VALUES ((SELECT id FROM staff WHERE telegram_id = %s), %s, 'transaction_cancelled', %s, %s)
            """, (user.telegram_id, client_id, audit_msg, points_to_revert))
            
            conn.commit()
            return {"status": "ok", "reverted": points_to_revert}
        except Exception as e:
            conn.rollback()
            raise HTTPException(status_code=500, detail="Ошибка при отмене")

@app.post("/api/admin/create-notification")
async def create_notification(request: Request, user: AuthUser = Depends(require_admin)):
    body = await request.json()
    expires_at = datetime.now(timezone.utc) + timedelta(days=body.get("days_valid", 7))

    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO notifications (type, title, description, image_url, expires_at)
            VALUES (%s, %s, %s, %s, %s) RETURNING id
        """, (body.get("type"), body.get("title"), body.get("description"), body.get("image_url"), expires_at))
        
        notif_id = cursor.fetchone()["id"]
        conn.commit()
        return {"id": notif_id, "status": "ok"}

@app.post("/api/admin/create-gift")
@limiter.limit("5/minute")
async def create_gift(request: Request, user: AuthUser = Depends(require_admin)):
    body = await request.json()
    name = body.get("name")
    points_cost = body.get("points_cost")
    image_url = body.get("image_url")

    if not name or not points_cost:
        raise HTTPException(status_code=400, detail="name and points_cost required")

    with get_db() as conn:
        cursor = conn.cursor()

        cursor.execute("SELECT id FROM gifts WHERE name = %s AND points_cost = %s", (name, points_cost))
        if cursor.fetchone():
            raise HTTPException(status_code=400, detail="Подарок уже существует")

        cursor.execute("""
            INSERT INTO gifts (name, points_cost, image_url, is_active) 
            VALUES (%s, %s, %s, true) 
            RETURNING id, name, points_cost, image_url
        """, (name, points_cost, image_url))

        gift = cursor.fetchone()

        if gift:
            audit_desc = f"Создан новый подарок: «{name}» за {points_cost} бонусов"
            cursor.execute("""
                INSERT INTO transactions (staff_id, type, description, target_type, target_id, points_change)
                VALUES (
                    (SELECT id FROM staff WHERE telegram_id = %s),
                    'gift_created',
                    %s,
                    'gift',
                    %s,
                    0
                )
            """, (user.telegram_id, audit_desc, gift['id']))

            conn.commit()

            asyncio.create_task(broadcast_new_gift(gift['name'], gift['points_cost']))

        return gift

@app.post("/api/admin/add-staff")
async def add_staff(request: Request, user: AuthUser = Depends(require_admin)):
    body = await request.json()
    t_id = body.get("telegram_id")
    name = body.get("name")
    role = body.get("role", "staff")

    with get_db() as conn:
        cursor = conn.cursor()
        # Если человек был клиентом — удаляем его из таблицы клиентов перед назначением сотрудником
        cursor.execute("DELETE FROM clients WHERE telegram_id = %s", (t_id,))
        
        cursor.execute("""
            INSERT INTO staff (telegram_id, name, role) VALUES (%s, %s, %s)
            ON CONFLICT (telegram_id) DO UPDATE SET name = %s, role = %s
        """, (t_id, name, role, name, role))
        conn.commit()
        return {"status": "ok"}

@app.post("/api/admin/staff-list")
@limiter.limit("5/minute")
async def get_all_staff(request: Request, user: AuthUser = Depends(require_admin)):
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT id, name, role FROM staff ORDER BY id")
        return cursor.fetchall()

@app.post("/api/admin/clients")
@limiter.limit("5/minute")
async def get_all_clients(request: Request, user: AuthUser = Depends(require_admin)):
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT card_number, first_name, last_name, points, total_earned_points, telegram_id, birth_date
            FROM clients ORDER BY total_earned_points DESC
        """)
        clients = []
        for row in cursor.fetchall():
            level = get_level(row["total_earned_points"])
            clients.append({
                "card_number": row["card_number"],
                "first_name": row["first_name"],
                "last_name": row["last_name"],
                "points": row["points"],
                "total_earned_points": row["total_earned_points"],
                "level": level,
                "telegram_id": row["telegram_id"],
                "birth_date": row["birth_date"]
            })
        return clients

@app.post("/api/admin/audit")
@limiter.limit("5/minute")
async def get_admin_audit(request: Request, user: AuthUser = Depends(require_admin)):
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT 
                t.id, 
                t.type,
                t.description, 
                t.created_at, 
                s.name AS staff_name
            FROM transactions t
            LEFT JOIN staff s ON t.staff_id = s.id
            WHERE t.type IN (
                'gift_deleted', 
                'gift_created', 
                'notification_created',  
                'notification_deleted', 
                'broadcast_sent'
            )
            ORDER BY t.created_at DESC
        """)
        return cursor.fetchall()

@app.post("/api/admin/delete-staff")
@limiter.limit("5/minute")
async def delete_staff(request: Request, user: AuthUser = Depends(require_admin)):
    body = await request.json()
    staff_id = body.get("staff_id")
    if not staff_id:
        raise HTTPException(status_code=400, detail="staff_id required")
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT role FROM staff WHERE id = %s", (staff_id,))
        staff = cursor.fetchone()
        if not staff:
            raise HTTPException(status_code=404, detail="Сотрудник не найден")
        if staff["role"] == "admin":
            raise HTTPException(status_code=403, detail="Нельзя удалять администраторов")
        cursor.execute("DELETE FROM staff WHERE id = %s", (staff_id,))
        conn.commit()
        return {"status": "ok"}

# Вспомогательная функция (если еще не добавил)
def rows_to_dict(cursor):
    columns = [desc[0] for desc in cursor.description]
    return [dict(zip(columns, row)) for row in cursor.fetchall()]

@app.post("/api/admin/transactions")
async def get_all_transactions_admin(request: Request, user: AuthUser = Depends(get_current_user)):
    # Проверка, что это не обычный клиент
    if user.role not in ["admin", "staff"]:
        raise HTTPException(status_code=403, detail="Access denied")

    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT 
                t.id, 
                t.type, 
                t.points_change, 
                t.description, 
                TO_CHAR(t.created_at, 'YYYY-MM-DD"T"HH24:MI:SS') as created_at,
                c.first_name || ' ' || c.last_name as client_name
            FROM transactions t
            JOIN clients c ON t.client_id = c.id
            ORDER BY t.created_at DESC 
            LIMIT 100
        """)
        return rows_to_dict(cursor)

# Получение всех уведомлений для админа (Стена новостей)
@app.post("/api/admin/all-notifications")
async def get_all_notifications_admin(user: AuthUser = Depends(get_current_user)):
    if user.role not in ["admin", "staff"]:
        raise HTTPException(status_code=403, detail="Доступ запрещен")

    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT 
                id, 
                type, 
                title, 
                description, 
                image_url, 
                TO_CHAR(created_at, 'YYYY-MM-DD"T"HH24:MI:SS') as created_at,
                TO_CHAR(expires_at, 'YYYY-MM-DD"T"HH24:MI:SS') as expires_at
            FROM notifications 
            ORDER BY created_at DESC
        """)
        return rows_to_dict(cursor)

# # Создание новой новости/новинки
# @app.post("/api/admin/broadcast")
# async def create_broadcast(request: Request, user: AuthUser = Depends(get_current_user)):
#     if user.role not in ["admin", "staff"]:
#         raise HTTPException(status_code=403, detail="Доступ запрещен")
    
#     data = await request.json()
#     # Ожидаем: title, message, type (news/promo), image_url
    
#     with get_db() as conn:
#         cursor = conn.cursor()
#         cursor.execute("""
#             INSERT INTO notifications (type, title, message, image_url, created_at, expires_at)
#             VALUES (%s, %s, %s, %s, NOW(), NOW() + interval '30 days')
#             RETURNING id
#         """, (data.get('type', 'news'), data.get('title'), data.get('message'), data.get('image_url')))
#         conn.commit()
#         return {"ok": True, "id": cursor.fetchone()[0]}

# === ТЕЛЕГРАМ WEBHOOK ===
@app.post("/webhook")
async def telegram_webhook(request: Request):
    data = await request.json()
    if "message" not in data: return {"ok": True}

    msg = data["message"]
    chat_id = msg["chat"]["id"]
    user_id = msg["from"]["id"]
    text = msg.get("text", "").lower()

    bot_token = os.getenv("BOT_TOKEN")
    web_app_url = "https://dwntwn-loyalty-frontend-io.vercel.app"

    response_text = ""
    reply_markup = None

    if text == "/start" or text == "/app":
        with get_db() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT 1 FROM clients WHERE telegram_id = %s", (user_id,))
            is_client = cursor.fetchone()
            cursor.execute("SELECT role FROM staff WHERE telegram_id = %s", (user_id,))
            is_staff = cursor.fetchone()

        if is_staff:
            response_text = f"👋 Привет! Панель управления <b>DWNTWN</b> доступна по кнопке ниже:"
        elif is_client:
            response_text = "☕️ Рады видеть вас снова! Ваша карта <b>DWNTWN</b> готова к использованию:"
        else:
            response_text = "☕️ Добро пожаловать в <b>DWNTWN</b>!\n\nЗаполните анкету, чтобы получить бонусную карту и подарок при первой покупке."
        
        reply_markup = {"inline_keyboard": [[{"text": "🎫 Открыть карту", "web_app": {"url": web_app_url}}]]}

    elif text == "/help":
        response_text = (
            "❓ <b>Помощь по программе DwnTwn</b>\n\n"
            "1. Нажмите кнопку «Открыть карту»\n"
            "2. Заполните анкету при первом входе\n"
            "3. Предъявляйте QR-код бариста при каждой покупке или скажите номер что указали при регистрации\n\n"
            "📩 <b>Поддержка:</b> @dwntwn_coffee_support_bot"
        )

    elif text == "/about":
        response_text = (
            "☕ <b>DwnTwn Loyalty</b>\n\n"
            "Эта программа — наша благодарность вам за то, что выбираете нас. "
            "Мы ценим вашу преданность и хотим радовать вас бонусами с каждой чашки!\n\n"
            "✨ <b>Главное о бонусах:</b>\n"
            "• Копите бонусы за покупки (от 1% до 10% в зависимости от уровня).\n"
            "• Обменивайте их на подарки из нашего каталога.\n"
            "<i>Подробные правила начисления уровней (IRON → PLATINA) доступны в приложении.</i>"
        )
    
    else:
        return {"ok": True}

    async with aiohttp.ClientSession() as session:
        payload = {
            "chat_id": chat_id,
            "text": response_text,
            "parse_mode": "HTML"
        }
        if reply_markup:
            payload["reply_markup"] = reply_markup

        await session.post(f"https://api.telegram.org/bot{bot_token}/sendMessage", json=payload)

    return {"ok": True}
# === HEALTH CHECK ===
@app.get("/health")
async def health_check():
    return {
        "status": "ok", 
        "timestamp": datetime.utcnow().isoformat(),
        "version": "1.0.2"
    }

class BroadcastRequest(BaseModel):
    title: str
    message: str
    link: Optional[str] = None
    image_url: Optional[HttpUrl] = None

@app.post("/api/admin/broadcast")
@limiter.limit("1/minute") # Рассылка — тяжелая операция, ограничим частоту запуска
async def send_broadcast(request: Request, user: AuthUser = Depends(require_admin)):
    body = await request.json()
    try:
        broadcast = BroadcastRequest(**body)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Ошибка валидации: {e}")

    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT telegram_id FROM clients")
        clients = cursor.fetchall()

    if not clients:
        return {"status": "ok", "sent_to": 0, "total": 0, "message": "Нет клиентов"}

    bot_token = os.getenv("BOT_TOKEN")
    
    sent_count = 0
    failed_count = 0

    base_text = f"📢 <b>{broadcast.title}</b>\n\n{broadcast.message}"
    if broadcast.link:
        base_text += f"\n\n<a href='{broadcast.link}'>Перейти по ссылке →</a>"

    async def run_broadcast():
        nonlocal sent_count, failed_count
        async with aiohttp.ClientSession() as session:
            for client in clients:
                t_id = client["telegram_id"]
                try:
                    # Telegram рекомендует не более 30 сообщений в секунду
                    await asyncio.sleep(0.05) 
                    
                    if broadcast.image_url:
                        payload = {
                            "chat_id": t_id,
                            "photo": str(broadcast.image_url),
                            "caption": base_text,
                            "parse_mode": "HTML"
                        }
                        url = f"https://api.telegram.org/bot{bot_token}/sendPhoto"
                    else:
                        payload = {
                            "chat_id": t_id,
                            "text": base_text,
                            "parse_mode": "HTML"
                        }
                        url = f"https://api.telegram.org/bot{bot_token}/sendMessage"

                    async with session.post(url, json=payload, timeout=aiohttp.ClientTimeout(total=5)) as resp:
                        if resp.status == 200:
                            sent_count += 1
                        else:
                            failed_count += 1
                except Exception:
                    failed_count += 1
        
        # Логируем результат в БД после завершения
        with get_db() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO transactions (staff_id, type, description, points_change)
                VALUES ((SELECT id FROM staff WHERE telegram_id = %s), 'broadcast_sent', %s, 0)
            """, (user.telegram_id, f"Рассылка «{broadcast.title}» завершена. Успешно: {sent_count}"))
            conn.commit()

    # Запускаем рассылку фоном, чтобы не заставлять админа ждать завершения HTTP-запроса
    asyncio.create_task(run_broadcast())

    return {
        "status": "started",
        "total_targets": len(clients),
        "info": "Рассылка запущена в фоновом режиме"
    }

# === СИСТЕМНЫЕ ФУНКЦИИ ===

def log_account_deletion(telegram_id: int):
    """Логирование удаления аккаунта для аудита"""
    try:
        with get_db() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO audit_log (user_id, action, details)
                VALUES (%s, 'account_deleted', %s)
            """, (telegram_id, "Аккаунт пользователя и данные анкеты удалены"))
            conn.commit()
    except Exception as e:
        logging.error(f"Audit log error: {e}")

# === ЗАПУСК ===

if __name__ == "__main__":
    import uvicorn
    # Render или другие хостинги передают порт через переменную окружения
    port = int(os.getenv("PORT", 8000))
    logging.info(f"🚀 DwnTwn Backend запущен на порту {port}")
    
    # Режим reload=True только для разработки (в env должен быть DEVELOPMENT=true)
    is_dev = os.getenv("ENVIRONMENT") == "development"
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=is_dev)