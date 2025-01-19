from database import engine, Base

from models.ProductModel import ProductModel
from models.SmartPhone import SmartPhone
from models.UsersModel import UsersModel



if __name__ == "__main__":
    print("Creating SQLite tables...")
    Base.metadata.create_all(bind=engine)
    print("Tables created successfully in SQLite database.")
