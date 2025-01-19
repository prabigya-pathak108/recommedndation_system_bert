from sqlalchemy import create_engine, Column, Integer, Text,String
from sqlalchemy.ext.declarative import declarative_base
from database import Base


class UsersModel(Base):
    __tablename__ = 'users'

    id = Column(Integer, primary_key=True)  # ID field
    age = Column(Integer, nullable=False)
    gender = Column(String(10), nullable=False)
    likes = Column(Text, nullable=False)  # Store likes as a comma-separated string
    dislikes = Column(Text, nullable=False)  # Store dislikes as a comma-separated string
    feedbacks = Column(Text, nullable=True)  # Store feedbacks as a comma-separated string