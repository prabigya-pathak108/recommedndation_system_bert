from sqlalchemy import create_engine, Column, Integer, Float, String, Text
from sqlalchemy.ext.declarative import declarative_base

from database import Base



class ProductModel(Base):
    __tablename__ = 'products'

    id = Column(Integer, primary_key=True, autoincrement=True)
    product_name = Column(String(400), nullable=False)
    product_category_tree = Column(Text, nullable=True)
    description = Column(Text, nullable=True)
    brand = Column(String(255), nullable=True)
    discounted_price = Column(Float, nullable=True)
    product_specifications = Column(Text, nullable=True)
    overall_rating = Column(String(255), nullable=True)

    def __repr__(self):
        return (f"<Product(id={self.id}, product_name='{self.product_name}', "
                f"brand='{self.brand}', discounted_price={self.discounted_price}, "
                f"overall_rating={self.overall_rating})>")

