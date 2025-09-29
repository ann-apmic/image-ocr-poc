from datetime import datetime
from typing import Union
from uuid import UUID

from pydantic import BaseModel


class Detail(BaseModel):
    detail: str


class IDMixin(BaseModel):
    id: Union[str, UUID, int]


class TokenMixin(BaseModel):
    token: str


class CreatedAtMixin(BaseModel):
    created_at: datetime


class UpdatedAtMixin(BaseModel):
    updated_at: datetime


class DatetimeMixin(CreatedAtMixin, UpdatedAtMixin):
    pass
