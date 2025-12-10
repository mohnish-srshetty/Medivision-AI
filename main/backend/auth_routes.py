from fastapi import HTTPException, Depends, status
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session
from typing import List
import db_models, schemas, auth, database

def register_auth_routes(app):
    """Register authentication and history routes"""
    
    @app.post("/auth/signup", response_model=schemas.Token)
    def signup(user: schemas.UserCreate, db: Session = Depends(database.get_db)):
        db_user = db.query(db_models.User).filter(db_models.User.email == user.email).first()
        if db_user:
            raise HTTPException(status_code=400, detail="Email already registered")
        
        hashed_password = auth.get_password_hash(user.password)
        new_user = db_models.User(email=user.email, full_name=user.full_name, hashed_password=hashed_password)
        db.add(new_user)
        db.commit()
        db.refresh(new_user)
        
        access_token = auth.create_access_token(data={"sub": new_user.email})
        return {"access_token": access_token, "token_type": "bearer"}

    @app.post("/auth/login", response_model=schemas.Token)
    def login(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(database.get_db)):
        user = db.query(db_models.User).filter(db_models.User.email == form_data.username).first()
        if not user or not auth.verify_password(form_data.password, user.hashed_password):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Incorrect username or password",
                headers={"WWW-Authenticate": "Bearer"},
            )
        access_token = auth.create_access_token(data={"sub": user.email})
        return {"access_token": access_token, "token_type": "bearer"}

    @app.get("/auth/me", response_model=schemas.User)
    def read_users_me(current_user: db_models.User = Depends(auth.get_current_user)):
        return current_user

    @app.get("/history", response_model=List[schemas.Report])
    def get_history(db: Session = Depends(database.get_db), current_user: db_models.User = Depends(auth.get_current_user)):
        return current_user.reports

    @app.post("/history", response_model=schemas.Report)
    def save_report(report: schemas.ReportCreate, db: Session = Depends(database.get_db), current_user: db_models.User = Depends(auth.get_current_user)):
        db_report = db_models.Report(**report.dict(), user_id=current_user.id)
        db.add(db_report)
        db.commit()
        db.refresh(db_report)
        return db_report
