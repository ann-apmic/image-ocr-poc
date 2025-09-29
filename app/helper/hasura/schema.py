from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, Field

from app.schema import IDMixin, TokenMixin


class HasuraEventCustomPayload(BaseModel):
    payload: TokenMixin


class HasuraEventReq(HasuraEventCustomPayload, BaseModel):
    scheduled_time: datetime
    name: str
    id: UUID


class HasuraModel(BaseModel):
    op: str
    data: dict


class HasuraModelTable(BaseModel):
    name: str


class HasuraModelPayload(IDMixin, BaseModel):
    event: HasuraModel
    table: HasuraModelTable


class HasuraAction(BaseModel):
    name: str


class HasuraAuth(BaseModel):
    user_id: str = Field("", alias="x-hasura-user-id")


class HasuraActionMixin(BaseModel):
    session_variables: HasuraAuth
    action: HasuraAction
