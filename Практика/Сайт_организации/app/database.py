from sqlalchemy import create_engine, Column, Integer, String
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

SQLALCHEMY_DATABASE_URL = "sqlite:///./ai_corp.db" # В проде заменить на PostgreSQL
engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class RequestLog(Base):
    __tablename__ = "logs"
    id = Column(Integer, primary_key=True, index=True)
    endpoint = Column(String, index=True)
    input_data = Column(String)
    output_data = Column(String)

Base.metadata.create_all(bind=engine)