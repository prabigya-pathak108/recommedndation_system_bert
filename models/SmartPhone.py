from sqlalchemy import create_engine, Column, Integer, Float, String, Boolean, MetaData
from sqlalchemy.ext.declarative import declarative_base

from database import Base

# Define the SmartPhone table
class SmartPhone(Base):
    __tablename__ = 'smartphone'
    
    id = Column(Integer, primary_key=True, autoincrement=True)  # Auto-increment ID
    brand_name = Column(String, nullable=False)
    model = Column(String, nullable=False)
    price = Column(Integer, nullable=False)
    avg_rating = Column(Float)
    is_5G = Column(Boolean, nullable=False)
    processor_brand = Column(String)
    num_cores = Column(Float)
    processor_speed = Column(Float)
    battery_capacity = Column(Float)
    fast_charging_available = Column(Boolean, nullable=False)
    fast_charging = Column(Float)
    ram_capacity = Column(Integer, nullable=False)
    internal_memory = Column(Integer, nullable=False)
    screen_size = Column(Float, nullable=False)
    refresh_rate = Column(Integer, nullable=False)
    num_rear_cameras = Column(Integer, nullable=False)
    os = Column(String)
    primary_camera_rear = Column(Float, nullable=False)
    primary_camera_front = Column(Float)
    extended_memory_available = Column(Boolean, nullable=False)
    resolution_height = Column(Integer, nullable=False)
    resolution_width = Column(Integer, nullable=False)

