import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder,MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from models.SmartPhone import SmartPhone

input_csv_file=r"C:\Users\prabigya\Desktop\work_here\Recommendo-Personalized-Shopping-Powered-by-LLMs\load_into_database\dataset\clean_smartphone_data.csv"
train_data= pd.read_csv(input_csv_file)

df=pd.read_csv(r"C:\Users\prabigya\Desktop\work_here\Recommendo-Personalized-Shopping-Powered-by-LLMs\models\scaled_data.csv")


def identify_similar_product(dict_of_model_brand,list_of_importatnt_features):
    numerical_cols = ["price", "avg_rating", "num_cores", "processor_speed", "battery_capacity",
                    "ram_capacity", "internal_memory", "screen_size", "refresh_rate",
                    "primary_camera_rear", "primary_camera_front", "resolution_height", "resolution_width"]
    
    
    feature_weights = {
        "price": 2.0,
        "avg_rating": 1.9,
        "ram_capacity": 1.9,
        "internal_memory": 1.7,
        "battery_capacity": 1.5,
        "screen_size": 0.6,
        "refresh_rate": 0.5,
        "resolution_width":0.05
    }
    keys_in_fw=list(feature_weights.keys())
    for imp_feat in list_of_importatnt_features:
        if imp_feat not in keys_in_fw:
            feature_weights[imp_feat]=1.3

    # Apply weights to numerical features
    for col in numerical_cols:
        weight = feature_weights.get(col, 1)
        df[col] = df[col] * weight
        
    list_of_all_product_models=[]
    
    for model in dict_of_model_brand["model"]:
        # Get reference device for the current model
        reference_device = df[df["model"] == model].drop(columns=["brand_name", "model"]).values

        # Compute similarity
        features = df.drop(columns=["brand_name", "model"]).values
        similarity_scores = cosine_similarity(reference_device, features)

        # Add similarity scores to the dataframe
        df["similarity"] = similarity_scores[0]
        df_sorted = df.sort_values(by="similarity", ascending=False)
        if (len(dict_of_model_brand["model"])==1):
            num_to_add = min(12 - len(list_of_all_product_models), 7)  # Limit to remaining slots or max 5
            most_similar = df_sorted.iloc[1 : 1 + num_to_add]["model"].tolist()
            list_of_all_product_models.extend(most_similar)
        else:
            # Add the most similar devices (excluding the query device itself)
            num_to_add = min(12 - len(list_of_all_product_models), 6)  # Limit to remaining slots or max 5
            most_similar = df_sorted.iloc[1 : 1 + num_to_add]["model"].tolist()
            list_of_all_product_models.extend(most_similar)
            list_of_all_product_models=list(set(list_of_all_product_models))

        # Break if we've filled the list
        if len(list_of_all_product_models) >= 12:
            return list_of_all_product_models
        
    # if len(list_of_all_product_models)<10:
    #     for brand in dict_of_model_brand["brands"]:
    #         # Filter devices by brand
    #         brand_devices = df[df["brand_name"] == brand]

    #         # Add up to the remaining slots
    #         num_to_add = min(10 - len(list_of_all_product_models), 5)
    #         if num_to_add > 0:
    #             most_similar = brand_devices.head(num_to_add)["model"].tolist()
    #             list_of_all_product_models.extend(most_similar)
    #             list_of_all_product_models=list(set(list_of_all_product_models))

    #         # Break if we've filled the list
    #         if len(list_of_all_product_models) >= 12:
    #             return list_of_all_product_models
    return list_of_all_product_models

def find_similar_product(dict_of_model_brand):
    obtained_list_of_product_models=identify_similar_product(dict_of_model_brand,[])
    filtered_df = df[df["model"].isin(obtained_list_of_product_models)]
    filtered_df["score"] = (
        0.3 * filtered_df["processor_speed"] +
        0.2 * filtered_df["ram_capacity"] +
        0.2 * filtered_df["battery_capacity"] +
        0.1 * filtered_df["primary_camera_rear"] +
        0.1 * filtered_df["screen_size"] -
        0.1 * filtered_df["price"]  # Subtract price for value-based ranking
    )

    # Rank devices based on score
    filtered_df_sorted = filtered_df.sort_values(by="score", ascending=False)
    final_product_list=list(filtered_df_sorted["model"])

    final_data=train_data[train_data["model"].isin(final_product_list)]
    return final_data[["model","brand_name","price","avg_rating"]]



from sqlalchemy import and_
from sqlalchemy.orm import sessionmaker

def apply_filter(session, model, filter_dict):
    """
    Apply dynamic filtering to the SQLAlchemy query based on the filter_dict,
    automatically inferring the datatype from the model.
    
    :param session: SQLAlchemy session object
    :param model: SQLAlchemy model class (e.g., User)
    :param filter_dict: Dictionary with filter conditions
    :return: Query object after applying filters
    """
    filters = []
    
    for column_name, condition in filter_dict.items():
        column = getattr(model, column_name)
        operator_type = condition.get("operator_type").lower()
        value = condition.get("value")
        
        # Automatically infer the datatype from the column's type
        column_type = str(column.type).lower().strip().replace("varchar","string").replace("nvarchar","string")
        # print("column_name is: ",column_name)
        # print("operator_type is: ",operator_type)
        # print("column_type is: ",column_type)
        # print("value is: ",value)

        # Apply different filter conditions based on the operator type
        if "string" in column_type and operator_type == "ilike":
            filters.append(column.ilike(f"%{value}%"))
        elif "string" in column_type and operator_type == "not_ilike":
            filters.append(~column.ilike(f"%{value}%"))
        elif "string" in column_type and operator_type == "in":
            if isinstance(value, list):
                filters.append(column.in_(value))
        elif "integer" in column_type and operator_type == "<":
            filters.append(column < value)
        elif "integer" in column_type and operator_type == ">":
            filters.append(column > value)
        elif "integer" in column_type and operator_type == "=":
            filters.append(column == value)
        elif "integer" in column_type and operator_type == "between":
            if isinstance(value, (list, tuple)) and len(value) == 2:
                filters.append(column.between(value[0], value[1]))
        elif "float" in column_type and operator_type == "<":
            filters.append(column < value)
        elif "float" in column_type and operator_type == ">":
            filters.append(column > value)
        elif "float" in column_type and operator_type == "=":
            filters.append(column == value)
        elif "float" in column_type and operator_type == "between":
            if isinstance(value, (list, tuple)) and len(value) == 2:
                filters.append(column.between(value[0], value[1]))
        elif "boolean" in column_type:
            # Boolean filters typically only use "="
            filters.append(column == (value in ["true", "True", True, "1", 1]))

    
    # print("Filter are: ",filters)

    # Apply the filters to the query
    query = session.query(model).filter(and_(*filters))
    return query



def fetch_smartphone(filter_dict,session_name,model= SmartPhone):
    final_list=[]
    filtered_query = apply_filter(session_name, model, filter_dict)
    # Execute the query and print the results
    products = filtered_query.all()

    for phone in products:
        final_list.append(f"ID: {phone.id}, Brand: {phone.brand_name}, Model: {phone.model}, Price: {phone.price}, "
            f"Avg Rating: {phone.avg_rating}, Is 5G: {phone.is_5G}, Processor: {phone.processor_brand}, "
            f"RAM: {phone.ram_capacity}, Storage: {phone.internal_memory}")
    return final_list
