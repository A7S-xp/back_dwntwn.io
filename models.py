# models.py
from pydantic import BaseModel, EmailStr, Field
from typing import Optional
from datetime import date, datetime

# === Клиенты ===
class ClientRegister(BaseModel):
    telegram_id: int
    first_name: str = Field(..., min_length=1, max_length=50)
    last_name: str = Field(..., min_length=1, max_length=50)
    email: Optional[EmailStr] = None
    phone: Optional[str] = Field(None, min_length=6, max_length=20)  # простая валидация
    birth_date: Optional[date] = None
    gender: Optional[str] = Field(None, pattern="^(male|female)$")

class ClientProfile(BaseModel):
    card_number: str
    first_name: str
    last_name: str
    points: int
    total_earned_points: int
    level: str
    telegram_id: int

# === Сотрудники ===
class StaffLogin(BaseModel):
    id: int
    name: str
    role: str  # 'staff' или 'admin'

# === Подарки ===
class GiftCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)
    points_cost: int = Field(..., gt=0)
    image_url: Optional[str] = None

class GiftResponse(GiftCreate):
    id: int

# === Уведомления ===
class NotificationCreate(BaseModel):
    type: str = Field(..., pattern="^(promotion|novelty|announcement)$")
    title: str = Field(..., min_length=1, max_length=100)
    description: str = Field(..., min_length=1)
    days_valid: int = Field(7, ge=1)
    image_url: Optional[str] = None

class NotificationResponse(BaseModel):
    id: int
    type: str
    title: str
    description: str
    image_url: Optional[str]
    expires_at: datetime  # FastAPI сам сериализует datetime → ISO string

# === Транзакции ===
class TransactionResponse(BaseModel):
    id: int
    client_name: str
    type: str
    points_change: int
    description: Optional[str]
    created_at: datetime