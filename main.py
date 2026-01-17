import os
import hmac
import hashlib
from urllib.parse import parse_qsl, unquote
import json
from datetime import datetime, timedelta, date
from contextlib import contextmanager
from pathlib import Path
from pydantic import BaseModel
import secrets
import string
import logging
from dotenv import load_dotenv
import asyncio
import aiohttp
from pydantic import BaseModel, HttpUrl
from typing import Optional
import uvicorn

from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter
from slowapi.util import get_remote_address
import html
import ipaddress

# Настройка логирования
logging.basicConfig(level=logging.INFO)

# Загружаем .env
load_dotenv(Path(__file__).parent.parent / ".env")

BOT_TOKEN = os.getenv("BOT_TOKEN")
if not BOT_TOKEN:
    raise ValueError("⚠️ BOT_TOKEN не найден в .env")

# === FastAPI APP ===
app = FastAPI(
    title="Система лояльности DwnTwn",
    description="Production-ready API для сети кофеен",
    version="1.0.0"
)

@app.on_event("startup")
async def startup_event():
    from database import get_db_connection
    from schemas import init_database
    conn = None
    try:
        conn = get_db_connection()
        init_database(conn)
        logging.info("✅ База данных успешно инициализирована")
    except Exception as e:
        logging.error(f"❌ Ошибка инициализации БД: {e}")
    finally:
        if conn: conn.close()

# === CORS — УБРАНЫ ПРОБЕЛЫ! ===
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

@app.middleware("http")
async def log_requests(request: Request, call_next):
    # Убираем query-параметры (там может быть initData)
    safe_url = str(request.url).split("?")[0]
    logging.info(f"Request: {request.method} {safe_url}")
    response = await call_next(request)
    return response

# === ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ===

def normalize_phone(phone: str | None) -> str | None:
    """
    Приводит номер к формату +79123456789.
    Возвращает None, если номер не передан.
    Выбрасывает ValueError, если номер передан, но некорректен.
    """
    if not phone:
        return None

    digits = ''.join(filter(str.isdigit, phone))

    if len(digits) == 11 and digits.startswith(('7', '8')):
        return f"+7{digits[1:]}"
    if len(digits) == 10:
        return f"+7{digits}"

    raise ValueError("Некорректный формат номера телефона")


def generate_card_number(conn) -> str:
    prefix = "DTLC"
    max_attempts = 10
    cursor = conn.cursor()
    for _ in range(max_attempts):
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


def send_user_notification(telegram_id: int, title: str, message: str):
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO user_notifications (telegram_id, title, message)
            VALUES (%s, %s, %s)
        """, (telegram_id, title, message))
        conn.commit()


# === ВАЛИДАЦИЯ TELEGRAM INITDATA ===
def validate_telegram_init_data(init_data: str, bot_token: str) -> dict:
    if not init_data or "hash=" not in init_data:
        raise HTTPException(status_code=401, detail="Invalid initData")
    try:
        parsed = dict(parse_qsl(init_data))
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid initData format")
    if "hash" not in parsed:
        raise HTTPException(status_code=401, detail="Missing hash")
    hash_ = parsed.pop("hash")
    data_check_string = "\n".join(f"{k}={v}" for k, v in sorted(parsed.items()))
    secret_key = hmac.new(b"WebAppData", bot_token.encode(), hashlib.sha256).digest()
    calculated_hash = hmac.new(secret_key, data_check_string.encode(), hashlib.sha256).hexdigest()
    if not hmac.compare_digest(calculated_hash, hash_):
        raise HTTPException(status_code=401, detail="Invalid Telegram data")
    return parsed

# Не рабочее((
# def validate_telegram_init_data(init_data: str, bot_token: str) -> dict:
#     if not init_data or "hash=" not in init_data:
#         raise HTTPException(status_code=401, detail="Invalid initData")

#     # 1. Разделяем на пары, НЕ парсим как query string
#     pairs = []
#     for part in init_data.split('&'):
#         if '=' in part:
#             k, v = part.split('=', 1)
#             pairs.append((k, v))  # v остаётся в исходном виде (с %7B...)

#     # 2. Извлекаем hash
#     hash_ = None
#     clean_pairs = []
#     for k, v in pairs:
#         if k == "hash":
#             hash_ = v
#         else:
#             clean_pairs.append((k, v))

#     if not hash_:
#         raise HTTPException(status_code=401, detail="Missing hash")

#     # 3. Формируем строку ДЛЯ ПОДПИСИ — значения НЕ ДЕКОДИРУЕМ!
#     data_check_string = "\n".join(f"{k}={v}" for k, v in sorted(clean_pairs))

#     # 4. Генерируем хэш
#     secret_key = hmac.new(b"WebAppData", bot_token.encode(), hashlib.sha256).digest()
#     calculated_hash = hmac.new(
#         secret_key,
#         data_check_string.encode(),
#         hashlib.sha256
#     ).hexdigest()

#     # 5. Сравниваем
#     if not hmac.compare_digest(calculated_hash, hash_):
#         raise HTTPException(status_code=401, detail="Invalid Telegram data")

#     # 6. Проверка срока действия (опционально, но рекомендуется)
#     auth_date_str = dict(clean_pairs).get("auth_date")
#     if auth_date_str:
#         try:
#             auth_date = int(auth_date_str)
#             if auth_date < int(datetime.utcnow().timestamp()) - 86400:
#                 raise HTTPException(status_code=401, detail="Init data expired")
#         except ValueError:
#             pass  # Игнорируем, если не число

#     # 7. Возвращаем исходные пары (для последующей обработки user)
#     return dict(clean_pairs)

def extract_telegram_id_from_init_data(init_data: str) -> int:
    parsed = validate_telegram_init_data(init_data, BOT_TOKEN)
    user_data_str = parsed.get("user")
    if not user_data_str:
        raise HTTPException(status_code=401, detail="User data missing in initData")
    try:
        user_json_str = unquote(user_data_str)
        user_dict = json.loads(user_json_str)
        return int(user_dict["id"])
    except (ValueError, KeyError, json.JSONDecodeError):
        raise HTTPException(status_code=401, detail="Invalid user data format")

# Не рабочее((
# def extract_telegram_id_from_init_data(init_data: str) -> int:
#     parsed = validate_telegram_init_data(init_data, BOT_TOKEN)
#     user_data_str = parsed.get("user")
#     if not user_data_str:
#         raise HTTPException(status_code=401, detail="User data missing in initData")
#     try:
#         # Декодируем ТОЛЬКО ПОСЛЕ ВАЛИДАЦИИ
#         user_json_str = unquote(user_data_str)  # %7B...%7D → {"id":...}
#         user_dict = json.loads(user_json_str)
#         return int(user_dict["id"])
#     except (ValueError, KeyError, json.JSONDecodeError) as e:
#         logging.error(f"Invalid user data: {user_data_str}, error: {e}")
#         raise HTTPException(status_code=401, detail="Invalid user data format")

async def send_telegram_message(telegram_id: int, text: str):
    """
    Отправляет HTML-сообщение пользователю в Telegram.
    """
    bot_token = os.getenv("BOT_TOKEN")
    if not bot_token:
        logging.error("BOT_TOKEN не задан — невозможно отправить сообщение")
        return

    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    payload = {
        "chat_id": telegram_id,
        "text": text,
        "parse_mode": "HTML"
    }

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload) as response:
                if response.status != 200:
                    error_data = await response.json()
                    # Игнорируем ошибку, если пользователь заблокировал бота
                    if error_data.get("error_code") != 403:
                        logging.warning(f"Не удалось отправить сообщение {telegram_id}: {error_data}")
    except Exception as e:
        logging.error(f"Ошибка отправки сообщения {telegram_id}: {e}")

async def broadcast_new_gift(gift_name: str, points_cost: int):
    """
    Рассылает уведомление всем зарегистрированным клиентам о новом подарке.
    """
    with get_db() as conn:
        cursor = conn.cursor()
        # Получаем ID всех клиентов
        cursor.execute("SELECT telegram_id FROM clients")
        users = cursor.fetchall()

    if not users:
        return

    text = (
        f"🎁 <b>У нас новый подарок!</b>\n\n"
        f"Теперь вы можете обменять баллы на: <b>{gift_name}</b>\n"
        f"Стоимость: <b>{points_cost}</b> баллов.\n\n"
        f"Заглядывайте в приложение, чтобы проверить свой баланс! ☕️"
    )

    for user in users:
        try:
            await send_telegram_message(user['telegram_id'], text)
            await asyncio.sleep(0.05) 
        except Exception as e:
            logging.error(f"Ошибка рассылки для {user['telegram_id']}: {e}")

async def send_welcome_message(telegram_id: int):
    bot_token = os.getenv("BOT_TOKEN")
    if not bot_token:
        logging.error("BOT_TOKEN не задан — невозможно отправить сообщение")
        return

    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    text = (
        "🎉 <b>Поздравляем!</b>\n\n"
        "Вы успешно зарегистрировались в программе лояльности <b>DwnTwn</b>!\n\n"
        "Теперь вы можете:\n"
        "• Накапливать бонусы за покупки\n"
        "• Обменивать бонусы на напитки\n"
        "• Участвовать в акциях\n\n"
        "💡 Используйте команды:\n"
        "/app — открыть карту\n"
    )
    payload = {
        "chat_id": telegram_id,
        "text": text,
        "parse_mode": "HTML"
    }

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload) as response:
                if response.status != 200:
                    logging.error(f"Не удалось отправить сообщение пользователю {telegram_id}: {await response.text()}")
    except Exception as e:
        logging.error(f"Ошибка при отправке сообщения пользователю {telegram_id}: {e}")


# === МОДЕЛИ ===
class AuthUser(BaseModel):
    telegram_id: int
    role: str  # 'client', 'staff', 'admin'


class ClientRegister(BaseModel):
    telegram_id: int
    first_name: str
    last_name: str
    phone: str | None = None
    email: str | None = None
    birth_date: str | None = None
    gender: str | None = None


# === ЗАВИСИМОСТИ ДЛЯ АВТОРИЗАЦИИ ===
async def get_current_user(request: Request) -> AuthUser:
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON")

    init_data = body.get("initData")
    if not init_data:
        raise HTTPException(status_code=401, detail="initData is required")

    telegram_id = extract_telegram_id_from_init_data(init_data)

    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT role FROM staff WHERE telegram_id = %s", (telegram_id,))
        staff = cursor.fetchone()
        if staff:
            return AuthUser(telegram_id=telegram_id, role=staff["role"])

        cursor.execute("SELECT 1 FROM clients WHERE telegram_id = %s", (telegram_id,))
        client = cursor.fetchone()
        if client:
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
async def check_registered(request: Request):
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
        logging.error(f"Check registered error: {e}")
        raise HTTPException(status_code=500, detail="Internal error")


@app.post("/api/client/register")
@limiter.limit("5/minute")
async def register_client(request: Request):
    body = await request.json()
    init_data = body.get("initData")
    if not init_data:
        raise HTTPException(status_code=400, detail="initData is required")

    telegram_id = extract_telegram_id_from_init_data(init_data)

    raw_phone = body.get("phone")
    try:
        normalized_phone = normalize_phone(raw_phone)
    except ValueError:
        raise HTTPException(status_code=400, detail="Некорректный формат номера телефона")

    client_data = ClientRegister(
        telegram_id=telegram_id,
        first_name=body.get("first_name", ""),
        last_name=body.get("last_name", ""),
        phone=normalized_phone,
        email=body.get("email"),
        birth_date=body.get("birth_date"),
        gender=body.get("gender")
    )

    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT id FROM clients WHERE telegram_id = %s", (client_data.telegram_id,))
        if cursor.fetchone():
            raise HTTPException(status_code=400, detail="Клиент уже зарегистрирован")

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

        asyncio.create_task(send_welcome_message(telegram_id))

        return {"card_number": card_number}


@app.post("/api/client/profile")
@limiter.limit("10/minute")
async def get_profile(request: Request, user: AuthUser = Depends(get_current_user)):
    if user.role != "client":
        raise HTTPException(status_code=403, detail="Client access required")
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT card_number, first_name, last_name, points, total_earned_points, birth_date
            FROM clients WHERE telegram_id = %s
        """, (user.telegram_id,))
        client = cursor.fetchone()
        if not client:
            raise HTTPException(status_code=404, detail="Клиент не найден")
        level = get_level(client["total_earned_points"])
        return {
            "card_number": client["card_number"],
            "first_name": client["first_name"],
            "last_name": client["last_name"],
            "points": client["points"],
            "total_earned_points": client["total_earned_points"],
            "level": level,
            "telegram_id": user.telegram_id,
            "birth_date": client["birth_date"]
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
@limiter.limit("10/minute")
async def get_gifts(request: Request, user: AuthUser = Depends(get_current_user)):
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT id, name, points_cost, image_url FROM gifts WHERE is_active = true ORDER BY points_cost")
        return cursor.fetchall()


@app.post("/api/client/delete-account")
@limiter.limit("5/minute")
async def delete_account(request: Request, user: AuthUser = Depends(get_current_user)):
    """
    Безопасное удаление аккаунта клиента и всех связанных данных.
    """
    if user.role != "client":
        raise HTTPException(status_code=403, detail="Only clients can delete their account")
    
    with get_db() as conn:
        cursor = conn.cursor()
        try:
            # 1. Сначала удаляем уведомления пользователя
            cursor.execute("DELETE FROM user_notifications WHERE telegram_id = %s", (user.telegram_id,))
            
            # 2. Получаем внутренний ID клиента для удаления транзакций
            cursor.execute("SELECT id FROM clients WHERE telegram_id = %s", (user.telegram_id,))
            client_row = cursor.fetchone()
            
            if client_row:
                client_id = client_row["id"]
                # 3. Удаляем все транзакции, связанные с этим клиентом
                cursor.execute("DELETE FROM transactions WHERE client_id = %s", (client_id,))
                
                # 4. Удаляем самого клиента
                cursor.execute("DELETE FROM clients WHERE id = %s", (client_id,))
            
            conn.commit()
            logging.info(f"User {user.telegram_id} deleted successfully.")
            
        except Exception as e:
            conn.rollback()
            logging.error(f"Error during account deletion for {user.telegram_id}: {e}")
            raise HTTPException(status_code=500, detail="Ошибка при удалении данных из базы")

    # Прощальное сообщение отправляем ПОСЛЕ удаления из БД
    farewell_text = (
        "🙏 <b>Спасибо, что пользовались нашей программой лояльности!</b>\n"
        "Ваши данные полностью удалены. Если захотите вернуться — мы всегда будем рады вам снова!\n"
        "До новых встреч в DWNTWN!"
    )
    # Используем create_task, чтобы не задерживать ответ пользователю
    asyncio.create_task(send_telegram_message(user.telegram_id, farewell_text))
    
    return {"status": "ok", "message": "Ваш аккаунт успешно удалён."}


# === СОТРУДНИКИ ===

@app.post("/api/staff/login")
@limiter.limit("5/minute")
async def staff_login(request: Request, user: AuthUser = Depends(require_staff)):
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT id, name, role FROM staff WHERE telegram_id = %s", (user.telegram_id,))
        staff = cursor.fetchone()
        return staff

@app.post("/api/staff/my-transactions")
@limiter.limit("10/minute")
async def get_staff_transactions(request: Request, user: AuthUser = Depends(require_staff)):
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT 
                t.id,
                CONCAT(c.first_name, ' ', c.last_name) AS client_name,
                t.points_change,
                t.description,
                t.created_at
            FROM transactions t
            JOIN clients c ON t.client_id = c.id
            WHERE t.staff_id = (SELECT id FROM staff WHERE telegram_id = %s)
            ORDER BY t.created_at DESC
            LIMIT 100
        """, (user.telegram_id,))
        rows = cursor.fetchall()
        # Преобразуем created_at в ISO-формат для JS
        result = []
        for row in rows:
            result.append({
                "id": row["id"],
                "client_name": row["client_name"],
                "points_change": row["points_change"],
                "description": row["description"],
                "created_at": row["created_at"].isoformat() if isinstance(row["created_at"], datetime) else str(row["created_at"])
            })
        return result

@app.post("/api/staff/client-by-card")
@limiter.limit("10/minute")
async def get_client_by_card(request: Request, user: AuthUser = Depends(require_staff)):
    body = await request.json()
    card_number = body.get("card_number")
    if not card_number:
        raise HTTPException(status_code=400, detail="card_number required")
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT id, telegram_id, first_name, last_name, points FROM clients WHERE card_number = %s", (card_number,))
        client = cursor.fetchone()
        if not client:
            raise HTTPException(status_code=404, detail="Клиент не найден")
        cursor.execute("SELECT total_earned_points FROM clients WHERE id = %s", (client["id"],))
        total_earned = cursor.fetchone()["total_earned_points"]
        level = get_level(total_earned)
        return {
            "id": client["id"],
            "name": f"{client['first_name']} {client['last_name']}",
            "points": client["points"],
            "level": level
        }


@app.post("/api/staff/client-by-phone")
@limiter.limit("10/minute")
async def get_client_by_phone(request: Request, user: AuthUser = Depends(require_staff)):
    body = await request.json()
    raw_phone = body.get("phone")
    if not raw_phone:
        raise HTTPException(status_code=400, detail="phone required")

    try:
        normalized = normalize_phone(raw_phone)
    except ValueError:
        raise HTTPException(status_code=400, detail="Некорректный номер")

    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT id, telegram_id, first_name, last_name, points, phone
            FROM clients
            WHERE phone = %s
        """, (normalized,))
        client = cursor.fetchone()

        if not client:
            raise HTTPException(status_code=404, detail="Клиент не найден")

        cursor.execute("SELECT total_earned_points FROM clients WHERE id = %s", (client["id"],))
        total_earned = cursor.fetchone()["total_earned_points"]
        level = get_level(total_earned)

        return {
            "id": client["id"],
            "telegram_id": client["telegram_id"],
            "name": f"{client['first_name']} {client['last_name']}",
            "points": client["points"],
            "level": level,
            "phone": client["phone"]
        }


@app.post("/api/staff/add-points")
@limiter.limit("10/minute")
async def add_points(request: Request, user: AuthUser = Depends(require_staff)):
    body = await request.json()
    client_id = body.get("client_id")
    purchase_amount = body.get("purchase_amount")
    if not client_id or not purchase_amount:
        raise HTTPException(status_code=400, detail="client_id and purchase_amount required")
    if purchase_amount > 2500:
        raise HTTPException(status_code=400, detail="Максимальная сумма покупки — 2500 руб.")
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT points, total_earned_points, telegram_id FROM clients WHERE id = %s", (client_id,))
        client = cursor.fetchone()
        if not client:
            raise HTTPException(status_code=404, detail="Клиент не найден")
        level = get_level(client["total_earned_points"])
        multiplier = {"PLATINA": 0.10, "GOLD": 0.07, "SILVER": 0.05, "BRONZE": 0.03, "IRON": 0.01}[level]
        points = max(1, int(purchase_amount * multiplier))
        new_points = client["points"] + points
        new_total = client["total_earned_points"] + points
        cursor.execute("UPDATE clients SET points = %s, total_earned_points = %s WHERE id = %s", (new_points, new_total, client_id))
        cursor.execute("""
            INSERT INTO transactions (client_id, staff_id, type, points_change, description)
            VALUES (%s, (SELECT id FROM staff WHERE telegram_id = %s), 'purchase', %s, %s)
        """, (client_id, user.telegram_id, points, f"Покупка на {purchase_amount} руб. (уровень {level})"))
        conn.commit()
        message_text = (
            f"🎉 <b>Бонусы начислены!</b>\n\n"
            f"Покупка на {purchase_amount} руб.\n"
            f"Начислено: <b>{points}</b> баллов.\n"
            f"Текущий баланс: <b>{new_points}</b> баллов."
        )
        asyncio.create_task(send_telegram_message(client["telegram_id"], message_text))
        return {"status": "ok", "new_points": new_points, "points_added": points, "level": level}

@app.post("/api/staff/redeem-gift")
@limiter.limit("10/minute")
async def redeem_gift(request: Request, user: AuthUser = Depends(require_staff)):
    body = await request.json()
    client_id = body.get("client_id")
    gift_id = body.get("gift_id")
    if not client_id or not gift_id:
        raise HTTPException(status_code=400, detail="client_id and gift_id required")
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT name, points_cost FROM gifts WHERE id = %s AND is_active = true", (gift_id,))
        gift = cursor.fetchone()
        if not gift:
            raise HTTPException(status_code=404, detail="Подарок недоступен")
        # ЗАПРАШИВАЕМ telegram_id
        cursor.execute("SELECT points, telegram_id FROM clients WHERE id = %s", (client_id,))
        client = cursor.fetchone()
        if not client or client["points"] < gift["points_cost"]:
            raise HTTPException(status_code=400, detail="Недостаточно баллов")
        new_points = client["points"] - gift["points_cost"]
        cursor.execute("UPDATE clients SET points = %s WHERE id = %s", (new_points, client_id))
        cursor.execute("""
            INSERT INTO transactions (client_id, staff_id, type, points_change, description)
            VALUES (%s, (SELECT id FROM staff WHERE telegram_id = %s), 'gift', %s, %s)
        """, (client_id, user.telegram_id, -gift["points_cost"], f"Подарок: {gift['name']}"))
        conn.commit()

        # ОТПРАВКА УВЕДОМЛЕНИЯ
        message_text = (
            f"🎁 <b>Подарок получен!</b>\n\n"
            f"Вы обменяли <b>{gift['points_cost']}</b> баллов на:\n"
            f"<b>{gift['name']}</b>.\n"
            f"Текущий баланс: <b>{new_points}</b> баллов."
        )
        asyncio.create_task(send_telegram_message(client["telegram_id"], message_text))

        return {"status": "ok", "gift_name": gift["name"], "new_points": new_points}

# === АДМИНКА ===

@app.post("/api/admin/gifts")
@limiter.limit("5/minute")
async def get_all_gifts(request: Request, user: AuthUser = Depends(require_admin)):
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT id, name, points_cost, image_url FROM gifts ORDER BY points_cost")
        return cursor.fetchall()

@app.post("/api/admin/delete-gift")
@limiter.limit("5/minute")
async def delete_gift(request: Request, user: AuthUser = Depends(require_admin)):
    body = await request.json()
    gift_id = body.get("gift_id")
    
    if not gift_id:
        raise HTTPException(status_code=400, detail="gift_id required")

    with get_db() as conn:
        cursor = conn.cursor()
        
        cursor.execute("SELECT name FROM gifts WHERE id = %s", (gift_id,))
        gift = cursor.fetchone()
        if not gift:
            raise HTTPException(status_code=404, detail="Подарок не найден")

        cursor.execute("DELETE FROM gifts WHERE id = %s", (gift_id,))
        
        audit_desc = f"Удален подарок: «{gift['name']}» (ID: {gift_id})"
        cursor.execute("""
            INSERT INTO transactions (staff_id, type, description, target_type, target_id, points_change)
            VALUES ((SELECT id FROM staff WHERE telegram_id = %s), 'gift_deleted', %s, 'gift', %s, 0)
        """, (user.telegram_id, audit_desc, gift_id))
        
        conn.commit()
        return {"status": "ok"}


@app.post("/api/admin/transactions")
@limiter.limit("5/minute")
async def get_transactions(request: Request, user: AuthUser = Depends(require_admin)):
    body = await request.json()
    start_date = body.get("start_date")
    end_date = body.get("end_date")
    with get_db() as conn:
        cursor = conn.cursor()
        query = """
            SELECT t.id, CONCAT(c.first_name, ' ', c.last_name) as client_name,
                   t.type, t.points_change, t.description, t.created_at
            FROM transactions t
            JOIN clients c ON t.client_id = c.id
            WHERE 1=1
        """
        params = []
        if start_date:
            query += " AND t.created_at >= %s"
            params.append(start_date)
        if end_date:
            query += " AND t.created_at <= %s"
            params.append(end_date)
        query += " ORDER BY t.created_at DESC"
        cursor.execute(query, params)
        return cursor.fetchall()

@app.post("/api/admin/create-notification")
@limiter.limit("5/minute")
async def create_notification(request: Request, user: AuthUser = Depends(require_admin)):
    body = await request.json()
    notif_type = body.get("type")
    title = body.get("title")
    description = body.get("description")
    image_url = body.get("image_url")
    days_valid = body.get("days_valid", 7)

    if not title or not description:
        raise HTTPException(status_code=400, detail="Title and description required")

    # Используем timezone-aware datetime для избежания проблем
    expires_at = datetime.utcnow() + timedelta(days=days_valid)

    with get_db() as conn:
        cursor = conn.cursor()
        
        # 1. Создаем уведомление
        cursor.execute("""
            INSERT INTO notifications (type, title, description, image_url, expires_at)
            VALUES (%s, %s, %s, %s, %s)
            RETURNING id
        """, (notif_type, title, description, image_url, expires_at))
        
        # БЕЗОПАСНОЕ ПОЛУЧЕНИЕ ID:
        row = cursor.fetchone()
        if isinstance(row, dict):
            notif_id = row['id']
        else:
            notif_id = row[0]

        # 2. Логируем действие
        audit_desc = f"Создано уведомление: «{title}» (тип: {notif_type})"
        
        cursor.execute("""
            INSERT INTO transactions (staff_id, type, description, target_type, target_id, points_change)
            VALUES (
                (SELECT id FROM staff WHERE telegram_id = %s),
                'notification_created',
                %s,
                'notification',
                %s,
                0
            )
        """, (user.telegram_id, audit_desc, notif_id))
        
        conn.commit()
        return {"status": "ok", "id": notif_id}

# @app.post("/api/admin/create-gift")
# @limiter.limit("5/minute")
# async def create_gift(request: Request, user: AuthUser = Depends(require_admin)):
#     body = await request.json()
#     name = body.get("name")
#     points_cost = body.get("points_cost")
#     image_url = body.get("image_url")
#     if not name or not points_cost:
#         raise HTTPException(status_code=400, detail="name and points_cost required")
#     with get_db() as conn:
#         cursor = conn.cursor()
#         cursor.execute("SELECT id FROM gifts WHERE name = %s AND points_cost = %s", (name, points_cost))
#         if cursor.fetchone():
#             raise HTTPException(status_code=400, detail="Подарок уже существует")
#         cursor.execute("INSERT INTO gifts (name, points_cost, image_url) VALUES (%s, %s, %s) RETURNING id, name, points_cost, image_url", (name, points_cost, image_url))
#         gift = cursor.fetchone()
#         conn.commit()
#         return gift

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


@app.post("/api/admin/staff-list")
@limiter.limit("5/minute")
async def get_all_staff(request: Request, user: AuthUser = Depends(require_admin)):
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT id, name, role FROM staff ORDER BY id")
        return cursor.fetchall()


@app.post("/api/admin/add-staff")
@limiter.limit("5/minute")
async def add_staff(request: Request, user: AuthUser = Depends(require_admin)):
    body = await request.json()
    telegram_id = body.get("telegram_id")
    name = body.get("name")
    role = body.get("role", "staff")
    if not telegram_id or not name:
        raise HTTPException(status_code=400, detail="telegram_id and name required")
    if role not in ("staff", "admin"):
        raise HTTPException(status_code=400, detail="Invalid role")
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT 1 FROM clients WHERE telegram_id = %s", (telegram_id,))
        if cursor.fetchone():
            cursor.execute("DELETE FROM clients WHERE telegram_id = %s", (telegram_id,))
        cursor.execute("""
            INSERT INTO staff (telegram_id, name, role)
            VALUES (%s, %s, %s)
            ON CONFLICT (telegram_id) DO UPDATE SET name = %s, role = %s
        """, (telegram_id, name, role, name, role))
        conn.commit()
        return {"status": "ok"}


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

@app.post("/api/admin/delete-notification")
@limiter.limit("10/minute")
async def delete_notification(request: Request, user: AuthUser = Depends(require_admin)):
    body = await request.json()
    notification_id = body.get("notification_id")
    if not notification_id:
        raise HTTPException(status_code=400, detail="notification_id required")

    with get_db() as conn:
        cursor = conn.cursor()
        # Получаем данные уведомления ДО удаления (для аудита)
        cursor.execute("""
            SELECT type, title, description FROM notifications WHERE id = %s
        """, (notification_id,))
        notif = cursor.fetchone()
        if not notif:
            raise HTTPException(status_code=404, detail="Уведомление не найдено")

        # Удаляем
        cursor.execute("DELETE FROM notifications WHERE id = %s", (notification_id,))
        
        # Логируем в аудит
        audit_desc = f"Удалено уведомление: [{notif['type']}] «{notif['title']}»"
        cursor.execute("""
            INSERT INTO transactions (staff_id, type, description, target_type, target_id, points_change)
            VALUES (
                (SELECT id FROM staff WHERE telegram_id = %s),
                'notification_deleted',
                %s,
                'notification',
                %s,
                0
            )
        """, (user.telegram_id, audit_desc, notification_id))
        
        conn.commit()
        return {"status": "ok"}

@app.post("/api/admin/all-notifications")
@limiter.limit("10/minute")
async def get_all_notifications(request: Request, user: AuthUser = Depends(require_admin)):
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT id, type, title, description, image_url, expires_at, created_at
            FROM notifications
            ORDER BY created_at DESC
        """)
        return cursor.fetchall()

# === ГОДОВЩИНА УЧАСТИЯ ===
@app.post("/api/internal/anniversary-check")
async def anniversary_check(request: Request):
    if request.client.host not in ("127.0.0.1", "::1"):
        raise HTTPException(status_code=403, detail="Forbidden")
    today = date.today()
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT id, telegram_id, first_name, last_name, created_at
            FROM clients
            WHERE EXTRACT(MONTH FROM created_at) = %s
              AND EXTRACT(DAY FROM created_at) = %s
              AND created_at <= %s
        """, (today.month, today.day, today - timedelta(days=365)))
        clients = cursor.fetchall()
        if not clients:
            return {"status": "ok", "message": "Нет годовщин сегодня"}
        results = []
        for client in clients:
            reg_date = client["created_at"].date()
            years = today.year - reg_date.year
            try:
                anniversary_this_year = reg_date.replace(year=today.year)
            except ValueError:
                anniversary_this_year = reg_date.replace(year=today.year, day=28)
            if anniversary_this_year == today and years >= 1:
                cursor.execute("""
                    UPDATE clients 
                    SET points = points + 100, total_earned_points = total_earned_points + 100
                    WHERE id = %s
                """, (client["id"],))
                cursor.execute("""
                    INSERT INTO transactions (client_id, type, points_change, description)
                    VALUES (%s, 'anniversary', 100, %s)
                """, (client["id"], f"Годовщина участия! {years} лет с нами!"))
                message_text = (
                    f"🎉 <b>Поздравляем с годовщиной!</b>\n\n"
                    f"Спасибо, что с нами уже {years} {'год' if years % 10 == 1 and years % 100 != 11 else 'года' if 2 <= years % 10 <= 4 and not (10 <= years % 100 <= 20) else 'лет'}!\n"
                    f"Вам начислено <b>100</b> бонусов!"
                )
                asyncio.create_task(send_telegram_message(client["telegram_id"], message_text))
                results.append({
                    "telegram_id": client["telegram_id"],
                    "name": f"{client['first_name']} {client['last_name']}",
                    "years": years
                })
        conn.commit()
        return {"status": "ok", "anniversaries": results}

@app.post("/webhook")
async def telegram_webhook(request: Request):
    # === 1. Проверка IP ===
    client_ip = request.client.host
    telegram_networks = ["149.154.160.0/20", "91.108.4.0/22"]
    if not any(ipaddress.ip_address(client_ip) in ipaddress.ip_network(net) for net in telegram_networks):
        return {"ok": False}

    try:
        data = await request.json()
        if "message" not in data:
            return {"ok": True}

        message = data["message"]
        chat_id = message["chat"]["id"]
        user = message.get("from", {})
        user_id = user.get("id")
        first_name = html.escape(user.get("first_name", "друг"))

        bot_token = os.getenv("BOT_TOKEN", "").strip()
        backend_url = os.getenv("BACKEND_URL", "https://back-dwntwn-io.onrender.com").strip().rstrip('/')
        web_app_url = "https://dwntwn-loyalty-frontend-io.vercel.app".strip()
        send_url = f"https://api.telegram.org/bot{bot_token}/sendMessage"

        # === 2. Обработка команд ===
        text = message.get("text", "").strip()

        if text == "/start":
            is_registered = False
            role = "client"

            # Проверка регистрации
            try:
                async with aiohttp.ClientSession() as session:
                    staff_resp = await session.post(f"{backend_url}/api/staff/login", json={"initData": f"user=%7B%22id%22%3A{user_id}%7D"})
                    if staff_resp.status == 200:
                        staff_data = await staff_resp.json()
                        role = staff_data.get("role", "client")
                        is_registered = True
                    else:
                        client_resp = await session.post(f"{backend_url}/api/client/check-registered", json={"telegram_id": user_id})
                        if client_resp.status == 200:
                            client_data = await client_resp.json()
                            is_registered = client_data.get("registered", False)
            except Exception as e:
                logging.warning(f"Ошибка проверки: {e}")

            # Кнопка Mini App
            app_button = {"text": "🎫 Открыть карту DwnTwn", "web_app": {"url": web_app_url}}
            inline_keyboard = {"inline_keyboard": [[app_button]]}

            if is_registered:
                msg = f"☕ Привет, {first_name}!\nРады видеть вас снова. Ваша карта доступна по кнопке ниже:"
            else:
                msg = (
                    f"☕ Привет, {first_name}!\n\n"
                    "🎉 Добро пожаловать в <b>DwnTwn</b>!\n\n"
                    "Нажмите кнопку ниже, чтобы войти в приложение и заполнить анкету участника:"
                )

            async with aiohttp.ClientSession() as session:
                await session.post(send_url, json={
                    "chat_id": chat_id, 
                    "text": msg, 
                    "parse_mode": "HTML",
                    "reply_markup": inline_keyboard
                })
            return {"ok": True}

        elif text == "/app":
            button = {"text": "🎫 Открыть лояльность", "web_app": {"url": web_app_url}}
            async with aiohttp.ClientSession() as session:
                await session.post(send_url, json={
                    "chat_id": chat_id,
                    "text": "📲 Ваша бонусная карта:",
                    "reply_markup": {"inline_keyboard": [[button]]}
                })

        elif text in ("/help", "/about"):
            text_map = {
                "/help": (
                    "❓ <b>Помощь по программе DwnTwn</b>\n\n"
                    "1. Нажмите кнопку «Открыть карту»\n"
                    "2. Заполните анкету при первом входе\n"
                    "3. Предъявляйте QR-код бариста при каждой покупке или скажите номер что указали при регистрации\n\n"
                    "📩 <b>Поддержка:</b> @dwntwn_coffee_support_bot"
                ),
                "/about": (
                    "☕ <b>DwnTwn Loyalty</b>\n\n"
                    "Эта программа — наша благодарность вам за то, что выбираете нас. "
                    "Мы ценим вашу преданность и хотим радовать вас бонусами с каждой чашки!\n\n"
                    "✨ <b>Главное о бонусах:</b>\n"
                    "• Копите бонусы за покупки (от 1% до 10% в зависимости от уровня).\n"
                    "• Обменивайте их на подарки из нашего каталога.\n"
                    "<i>Подробные правила начисления уровней (IRON → PLATINA) доступны в приложении.</i>"
                )
            }
            async with aiohttp.ClientSession() as session:
                await session.post(send_url, json={
                    "chat_id": chat_id, 
                    "text": text_map.get(text, ""), 
                    "parse_mode": "HTML"
                })
            return {"ok": True}

        return {"ok": True}

    except Exception as e:
        logging.error(f"Ошибка: {e}")
        return {"ok": False}


# === HEALTH CHECK ===
@app.get("/health")
async def health_check():
    return {"status": "ok", "timestamp": datetime.utcnow().isoformat()}
class BroadcastRequest(BaseModel):
    title: str
    message: str
    link: Optional[str] = None
    image_url: Optional[HttpUrl] = None  # Валидация URL

@app.post("/api/admin/broadcast")
@limiter.limit("3/minute")
async def send_broadcast(request: Request, user: AuthUser = Depends(require_admin)):
    body = await request.json()
    try:
        broadcast = BroadcastRequest(**body)  # ← валидация URL здесь
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid input: {e}")

    # Получаем всех клиентов
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT telegram_id FROM clients")
        clients = cursor.fetchall()

    if not clients:
        return {"status": "ok", "sent_to": 0, "total": 0, "message": "Нет клиентов для рассылки"}

    bot_token = os.getenv("BOT_TOKEN")
    if not bot_token:
        raise HTTPException(status_code=500, detail="BOT_TOKEN не настроен")

    sent_count = 0
    failed_ids = []

    base_text = f"📢 <b>{broadcast.title}</b>\n{broadcast.message}"
    if broadcast.link:
        base_text += f"\n<a href='{broadcast.link}'>Подробнее</a>"

    async with aiohttp.ClientSession() as session:
        for client in clients:
            telegram_id = client["telegram_id"]
            try:
                if broadcast.image_url:
                    # Отправляем фото + подпись
                    payload = {
                        "chat_id": telegram_id,
                        "photo": str(broadcast.image_url),
                        "caption": base_text,
                        "parse_mode": "HTML"
                    }
                    url = f"https://api.telegram.org/bot{bot_token}/sendPhoto"
                else:
                    # Отправляем текст
                    payload = {
                        "chat_id": telegram_id,
                        "text": base_text,
                        "parse_mode": "HTML",
                        "disable_web_page_preview": False
                    }
                    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"

                async with session.post(url, json=payload, timeout=aiohttp.ClientTimeout(total=5)) as resp:
                    if resp.status == 200:
                        sent_count += 1
                    else:
                        error_data = await resp.json()
                        if error_data.get("error_code") == 403:
                            logging.info(f"Пользователь {telegram_id} заблокировал бота")
                        else:
                            failed_ids.append(telegram_id)
                            logging.warning(f"Ошибка отправки {telegram_id}: {error_data}")
            except Exception as e:
                logging.error(f"Исключение при отправке {telegram_id}: {e}")
                failed_ids.append(telegram_id)

    # Логируем в аудит
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO transactions (staff_id, type, description, points_change)
            VALUES (
                (SELECT id FROM staff WHERE telegram_id = %s),
                'broadcast_sent',
                %s,
                0
            )
        """, (
            user.telegram_id,
            f"Пуш-рассылка: «{broadcast.title}» (доставлено: {sent_count}/{len(clients)})"
        ))
        conn.commit()

    return {
        "status": "ok",
        "sent_to": sent_count,
        "total": len(clients),
        "failed": len(failed_ids)
    }

def log_account_deletion(telegram_id: int):
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO audit_log (user_id, action, details)
            VALUES (%s, 'account_deleted', %s)
        """, (telegram_id, "Аккаунт удалён"))
        conn.commit()

# === ИНИЦИАЛИЗАЦИЯ БАЗЫ ДАННЫХ ПРИ СТАРТЕ ===
# def initialize_database():
#     from database import get_db_connection
#     from schemas import init_database
#     try:
#         conn = get_db_connection()
#         init_database(conn)
#         logging.info("✅ База данных успешно инициализирована")
#     except Exception as e:
#         logging.error(f"❌ Ошибка инициализации БД: {e}")
#         raise
#     finally:
#         conn.close()

# initialize_database()


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    logging.info(f"🚀 Запуск сервера на порту {port}")
    uvicorn.run("main:app", host="0.0.0.0", port=port)