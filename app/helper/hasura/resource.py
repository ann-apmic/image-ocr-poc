from fastapi import Header, HTTPException, status

from .schemas import HasuraEventReq, HasuraModelPayload

HASURA_KEY = "hasura_Event_trigger"


async def auth_hasura_events(hasura_event_req: HasuraEventReq):
    """驗證 Hasura 來源"""
    if hasura_event_req.payload.token != HASURA_KEY:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED)
    return hasura_event_req


async def auth_hasura_model(
    hasura_model_payload: HasuraModelPayload,
    token: str = Header(...),
) -> HasuraModelPayload:
    """驗證 hasura 資料列更新"""
    if token != HASURA_KEY:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED)
    return hasura_model_payload
