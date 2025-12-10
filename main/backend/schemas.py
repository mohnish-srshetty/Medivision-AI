from pydantic import BaseModel
from typing import Optional, List
from datetime import datetime

# User Schemas
class UserBase(BaseModel):
    email: str

class UserCreate(UserBase):
    full_name: str
    password: str

class UserLogin(UserBase):
    password: str

class User(UserBase):
    id: int
    full_name: str
    created_at: datetime

    class Config:
        from_attributes = True

# Token Schemas
class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    email: Optional[str] = None

# Report Schemas
class ReportBase(BaseModel):
    modality: str
    disease_detected: str
    confidence_score: float
    report_text: str
    image_path: Optional[str] = None

class ReportCreate(ReportBase):
    pass

class Report(ReportBase):
    id: int
    user_id: int
    created_at: datetime

    class Config:
        orm_mode = True
