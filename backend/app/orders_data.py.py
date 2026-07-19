import json
from app.config import ORDERS_PATH

ORDERS: dict = json.loads(ORDERS_PATH.read_text(encoding="utf-8"))