from core.config import SECRET_KEY
from fastapi import Header, HTTPException, status


def verify_token(token: str = Header(...)):
    if token != SECRET_KEY:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Token invalid")
