from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from app.auth import get_user, login

router = APIRouter()

class LoginRequest(BaseModel):
    username: str
    password: str

class LoginResponse(BaseModel):
    token: str
    display_name: str
    role: str

@router.post("/login", response_model=LoginResponse)
def login_route(request: LoginRequest) -> LoginResponse:
    token = login(request.username, request.password)
    if not token:
        raise HTTPException(status_code=401, detail="Invalid username or password")
    user = get_user(token)
    assert user is not None
    return LoginResponse(token=token, display_name=user.display_name, role=user.role)
    
    