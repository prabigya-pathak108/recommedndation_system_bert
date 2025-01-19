import os
import json

from services.find_similar import find_similar_product
from langchain.prompts import PromptTemplate
from pydantic import BaseModel
from typing import List
from pydantic import BaseModel
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel
from typing import List, Dict, Any

from difflib import SequenceMatcher
import re


from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import LLMChain


class CategoriesOutputClass(BaseModel):
    categories: List[str] = []  # Default to an empty list



def chain_to_find_categories_of_product(api_key_used,model_used,output_parser=PydanticOutputParser(pydantic_object=CategoriesOutputClass)):
    prompt_template = PromptTemplate(
    input_variables=["list_of_catrgories","user_query"],
    partial_variables={"format_instructions": output_parser.get_format_instructions()},
    template="""
        The list of categories is:
        {list_of_catrgories}

        Use above list and generate a list containing categories(must be same and present in above list) of product about which user is asking.

        User Question:
        {user_query}

        "{format_instructions}"
    """
)

    llm = ChatGoogleGenerativeAI(api_key=api_key_used,
        model=model_used,
        temperature=0)
    return LLMChain(llm=llm, prompt=prompt_template)


class ListOfImportantkeywords(BaseModel):
    important_keywords: List[str] = []  # Default to an empty list


def chain_to_find_important_keywords(api_key_used,model_used,output_parser=PydanticOutputParser(pydantic_object=ListOfImportantkeywords)):
    prompt_template = PromptTemplate(
    input_variables=["user_query"],
    partial_variables={"format_instructions": output_parser.get_format_instructions()},
    template="""
        You are an expert in finding keywords that are important in finding keywords that are important in SQL building from user query.

        Human Query:
        {user_query}

        "{format_instructions}"
    """
)

    llm = ChatGoogleGenerativeAI(api_key=api_key_used,
        model=model_used,
        temperature=0)
    return LLMChain(llm=llm, prompt=prompt_template)



class FilterCondition(BaseModel):
    value: Any
    operator_type: str

class DictofRelatedColumns(BaseModel):
    filters: Dict[str, FilterCondition]


def chain_to_obtain_columns(api_key_used,model_used,output_parser=PydanticOutputParser(pydantic_object=DictofRelatedColumns)):
    prompt_template = PromptTemplate(
    input_variables=["table_schema","table_sample_input","user_query"],
    partial_variables={"format_instructions": output_parser.get_format_instructions()},
    template="""
        Analyze user query. Use table schema as reference and create a dictionary use column name of table schema as key, obtained keyword in user query as value and what operation like (>,<,=,ILIKE,+,IN, BETWEEN,etc) needs to be applied in operator_type.
        Table Schema containing column name and description:
        {table_schema}

        The sample rows in table is:
        {table_sample_input}

        Human Query:
        {user_query}

        "{format_instructions}"
    """
)

    llm = ChatGoogleGenerativeAI(api_key=api_key_used,
        model=model_used,
        temperature=0)
    return LLMChain(llm=llm, prompt=prompt_template)


def chain_to_obtain_columns_with_generating_sql_first(api_key_used,model_used,output_parser=PydanticOutputParser(pydantic_object=DictofRelatedColumns)):
    prompt_template = PromptTemplate(
    input_variables=["table_schema","user_query"],
    partial_variables={"format_instructions": output_parser.get_format_instructions()},
    template="""
        Step 1: Generate SQL Query
        Based on the table schema and the user's query, construct a valid SQL query. Use operators like >,<,=,ILIKE,+,IN,etc. BETWEEN The SQL should retrieve the necessary data while adhering to the schema and logical constraints from the query.

        Table Schema (columns and descriptions):
        {table_schema}

        User Query:
        {user_query}

        Generated SQL Query:
        - Write the SQL query based on the above schema and query.

        Step 2: Analyze SQL for Associated Columns and Values
        After generating the SQL query, analyze it to identify the columns involved and their associated values. Provide a dictionary where:
        - The key is the column name from the table schema.
        - The value contains the corresponding condition(s) (e.g., operator and value) applied to that column.

        Output Format:
        "{format_instructions}"

        Donot give any explanation. Just give dict.
        """
    )

    llm = ChatGoogleGenerativeAI(api_key=api_key_used,
        model=model_used,
        temperature=0)
    return LLMChain(llm=llm, prompt=prompt_template)

def find_matching_products_model_and_brand(query, list_of_models_product, threshold=0.8):
    query=query.lower()
    brand_mapping = {
    'apple': ['apple', 'iphone'],
    'gionee': ['gionee'],
    'honor': ['honor'],
    'huawei': ['huawei'],
    'oneplus': ['oneplus'],
    'poco': ['poco'],
    'realme': ['realme'],
    'redmi': ['redmi'],
    'samsung': ['samsung'],
    'vivo': ['vivo'],
    'xiaomi': ['xiaomi']}
    def get_brand_from_query(query):
        query = query.lower()
        for brand, aliases in brand_mapping.items():
            if query in aliases:
                return brand
    if threshold ==0.9:
        return get_brand_from_query(query=query)
    
    def clean_string(text):
        return re.sub(r'[^\w\s]', '', str(text)).lower().strip()
    
    def get_base_model(model_name):
        return re.sub(r'\([^)]*\)|\d+GB|\d+\s*GB', '', model_name).strip()
    
    def similarity_score(str1, str2):
        return SequenceMatcher(None, str1, str2).ratio()
    
    cleaned_query = clean_string(query)
    base_query = get_base_model(cleaned_query)
    direct_matches = [value for value in list_of_models_product if query in value.lower()]
    matches = []
    if len(direct_matches) != 0:
        best_score = -100  # Initialize best score as a negative value
        best_product = None  # Initialize the best product variable
        for product in direct_matches:
            cleaned_product_name = clean_string(product)
            base_product_name = get_base_model(cleaned_product_name)

            # Calculate similarity for both full name and base model
            full_similarity = similarity_score(cleaned_query, cleaned_product_name)
            base_similarity = similarity_score(base_query, base_product_name)

            # Use the higher of the two scores
            current_best_score = max(full_similarity, base_similarity)

            if current_best_score > best_score:
                best_score = current_best_score
                best_product = product
        return [best_product]
    
    matches = []
    for product in list_of_models_product:
        cleaned_product_name = clean_string(product)
        base_product_name = get_base_model(cleaned_product_name)
        
        # Calculate similarity for both full name and base model
        full_similarity = similarity_score(cleaned_query, cleaned_product_name)
        base_similarity = similarity_score(base_query, base_product_name)
        
        # Use the higher of the two scores
        best_score = max(full_similarity, base_similarity)
        
        if best_score >= threshold:
            matches.append((product, best_score))
    
    # Sort by similarity score in descending order
    sorted_matches = sorted(matches, key=lambda x: x[1], reverse=True)

    # Return only the sorted products
    return [product for product, score in sorted_matches[:2]]
