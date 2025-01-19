from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.orm import Session


Base = declarative_base()
DATABASE_URL = "sqlite:///./products.db"  # Relative path

# SQLite engine setup
engine = create_engine(DATABASE_URL, echo=True)

# Session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def get_db():
    db=SessionLocal()
    try:
        yield db
    finally:
        print("----db_closed___")
        db.close()