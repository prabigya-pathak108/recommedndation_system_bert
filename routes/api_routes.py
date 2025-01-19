from fastapi import Depends,APIRouter,UploadFile, File,HTTPException, Request,BackgroundTasks,Query
from services.llm_chains import get_related_products
from langchain.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pydantic import BaseModel, ConfigDict, Field, SecretStr
from langchain_core.embeddings import Embeddings
from typing import Any, Dict, List, Optional
from sentence_transformers import SentenceTransformer
from sqlalchemy.orm import Session
from sqlalchemy.exc import SQLAlchemyError
from fastapi import Body

from models.UsersModel import UsersModel
from models.ProductModel import ProductModel
from models.SmartPhone import SmartPhone

from database import get_db  

from pydantic import BaseModel
from typing import Optional

class ProductSuggestion(BaseModel):
    id: int
    categories: str
    brand: str
    name: Optional[str] = None
    description: Optional[str] = None


class FineTunedHuggingFaceModel(Embeddings):
    def __init__(self, model_name: str):
        """Initialize with the model name or path."""
        self.model_name = model_name
        self.model = SentenceTransformer(model_name, device="cpu")

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        # Get the embeddings for the documents
        embeddings = self.model.encode(texts, convert_to_tensor=True)

        # Convert the tensor embeddings into a list of lists (each inner list represents an embedding)
        return embeddings.cpu().numpy().tolist()

    def embed_query(self, text: str) -> List[float]:
        return self.embed_documents([text])[0]
    

model_dir=r"C:\Users\prabigya\Desktop\work_here\Recommendo-Personalized-Shopping-Powered-by-LLMs\bge_models_weights"
fine_tuned_model = SentenceTransformer(
    model_dir, device="cpu")
embeddings=FineTunedHuggingFaceModel(model_name=model_dir)


router = APIRouter()

@router.get("/recommendation/")
async def get_recommendation(
    user_id: str = Query(..., description="The ID of the user"),local_kw: str = Query(None, description="Optional local keyword"), db: Session = Depends(get_db)
):
    try:
        # Fetch the user from the database using the user_id
        user = db.query(UsersModel).filter(UsersModel.id == user_id).first()

        # Check if the user exists
        if user is None:
            raise HTTPException(status_code=404, detail="User not found")

        # Get likes and dislikes as lists
        likes = user.likes.split(",") if user.likes else []
        dislikes = user.dislikes.split(",") if user.dislikes else []

        return {
            "user_id": user.id,
            "likes": likes,
            "dislikes": dislikes
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")
    

@router.get("/search_product/")
async def search_product(
    query: str = Query(..., description="The product to be searched")
):
    try:
        result=get_related_products(query)
        return {
            "user_id": result,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")
    
@router.post("/suggestion/")
async def post_suggestion(
    user_id: str = Query(..., description="The ID of the user"),
    feedback: str = Query(..., description="The feedback of the user"),
    product: ProductSuggestion = Body(..., description="The product details"),  # Use Pydantic model
    db: Session = Depends(get_db),
):
    try:
        # Fetch the user from the database using the user_id
        user = db.query(UsersModel).filter(UsersModel.id == user_id).first()

        # Check if the user exists
        if user is None:
            raise HTTPException(status_code=404, detail="User not found")

        feedbacks = user.feedbacks.strip() if user.feedbacks else ""
        appended_feedback = feedbacks + " " + feedback

        # Process the product data
        product_data = {
            "id": product.id,
            "categories": product.categories,
            "brand": product.brand,
            "name": product.name,
            "description": product.description,
        }

        # Example: Save feedback and product data to the database (not implemented)
        # db.add(FeedbackModel(user_id=user.id, product_id=product.id, feedback=feedback))
        # db.commit()

        return {
            "success": "Feedback Submitted",
            "product": product_data,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")