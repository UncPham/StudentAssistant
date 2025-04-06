from fastapi import APIRouter, Request, UploadFile, File, Depends, Body 
# RAG
from unstructured.partition.pdf import partition_pdf
import base64
# from IPython.display import Image, display
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import uuid
from langchain.vectorstores import Chroma
from langchain.storage import InMemoryStore
from langchain.schema.document import Document
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.schema import HumanMessage
import tempfile
import shutil
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from huggingface_hub import login
import weaviate
from dotenv import load_dotenv
import os
import re
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from backend.config import GROQ_API_KEY, HUGGINGFACE_API_KEY
from transformers import AutoModelForCausalLM, AutoTokenizer

# import logging

from pydantic import BaseModel

class QueryRequest(BaseModel):
    query: str
    file: str

router = APIRouter()

load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

def get_db(request: Request):
    return request.app.weaviate_client

def setup_weaviate_schema(client: weaviate.WeaviateClient, class_name: str):
    schema = {
        "class": class_name,
        "properties": [
            {"name": "doc_id", "dataType": ["string"]},
            {"name": "summary", "dataType": ["text"]},
            {"name": "original_text", "dataType": ["text"]},
            {"name": "page_number", "dataType": ["int"]}
        ],
        "vectorizer": "none"
    }
    # Kiểm tra xem class đã tồn tại chưa
    existing_classes = client.collections.list_all()
    if class_name not in [collection.name for collection in existing_classes.values()]:
        client.collections.create_from_dict(schema)

@router.post("/embeddings")
async def embeddings_pdf(file: UploadFile = File(...), weaviate_client=Depends(get_db)):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        shutil.copyfileobj(file.file, temp_file)
        temp_file_path = temp_file.name 

    # login(token=HUGGINGFACE_API_KEY)
    
    chunks = partition_pdf(
        filename=temp_file_path,
        infer_table_structure=True,            # extract tables
        strategy="hi_res",                     # mandatory to infer tables

        # extract_image_block_types=["Image"],   # Add 'Table' to list to extract image of tables
        # image_output_dir_path=output_path,   # if None, images and tables will saved in base64

        # extract_image_block_to_payload=True,   # if true, will extract base64 for API usage

        chunking_strategy="by_title",          # or 'basic'
        max_characters=10000,                  # defaults to 500
        combine_text_under_n_chars=2000,       # defaults to 0
        new_after_n_chars=6000,
        include_page_breaks=True

        # extract_images_in_pdf=True,          # deprecated
    )
    
    # tables = []
    texts = []
    page_numbers = []
    for chunk in chunks:
        texts.append(str(chunk))
        page_numbers.append(chunk.metadata.page_number if hasattr(chunk.metadata, 'page_number') else 1)

    # Kiểm tra xem texts có rỗng không
    if not texts:
        os.remove(temp_file_path)
        return {
            "status": "error",
            "message": "Không tìm thấy nội dung văn bản trong PDF để xử lý",
        }
    else:
        print("Found texts")

    prompt_text = """
    You are an assistant tasked with summarizing tables and text.
    Give a concise summary of the table or text.

    Respond only with the summary, no additionnal comment.
    Do not start your message by saying "Here is a summary" or anything like that.
    Just give the summary as it is.

    Table or text chunk: {element}

    """
    prompt = ChatPromptTemplate.from_template(prompt_text)

    # Summary chain
    model = ChatGroq(temperature=0.5, model="llama-3.1-8b-instant")
    summarize_chain = {"element": lambda x: x} | prompt | model | StrOutputParser()

    text_summaries = summarize_chain.batch(texts)

    # Kiểm tra xem text_summaries có rỗng không
    if not text_summaries:
        os.remove(temp_file_path)
        return {
            "status": "error",
            "message": "Không thể tạo bản tóm tắt từ nội dung PDF",
        }
    else:
        print("Found text_summaries")

    embedding_function = HuggingFaceEmbeddings()
    summary_embeddings = embedding_function.embed_documents(text_summaries)

    
    class_name = f"Vector_Store_{file.filename.replace('.', '_')}"
    class_name = re.sub(r'[^a-zA-Z0-9_]', '_', class_name)
    setup_weaviate_schema(weaviate_client, class_name)

    collection = weaviate_client.collections.get(class_name)
    doc_ids = [str(uuid.uuid4()) for _ in texts]

    for i, (text, summary, embedding, page) in enumerate(zip(texts, text_summaries, summary_embeddings, page_numbers)):
        doc_id = doc_ids[i]
        doc_data = {
            "doc_id": doc_id,
            "summary": summary,
            "original_text": text,
            "page_number": page
        }
        collection.data.insert(
            properties=doc_data,
            vector=embedding
        )

    # Xóa file tạm
    os.remove(temp_file_path)

    return {
        "status": "success",
        "message": "Successfully uploaded and processed PDF",
    }

# Hàm xây dựng prompt
def build_prompt(kwargs):
    docs = kwargs["context"]  # Danh sách các tài liệu từ Weaviate
    user_question = kwargs["question"]

    # Tạo context từ các tài liệu trả về, với số thứ tự
    if docs:
        context_text = "\n".join(
            [f"[{i+1}] Summary: {doc['summary']}\n" for i, doc in enumerate(docs)]
        )
    else:
        context_text = "No context available."

    # Tạo template prompt
    prompt_template = f"""
    You are a technical expert.
    You answer the questions truthfully on the basis of the documents provided.
    For each document, check whether it is related to the question.
    To answer the question, only use documents that are related to the question.
    Ignore documents that do not relate to the question.
    If the answer is contained in several documents, summarize them.
    Always use references in the form [NUMBER OF DOCUMENT] if you use information from a document, e.g. [3] for document [3].
    Never name the documents, only enter a number in square brackets as a reference.
    The reference may only refer to the number in square brackets after the passage.
    Otherwise, do not use brackets in your answer and give ONLY the number of the document without mentioning the word document.
    Give a precise, accurate and structured answer without repeating the question.
    Answer only on the basis of the documents provided. Do not make up facts.
    If the documents cannot answer the question or you are not sure, say so.
    These are the documents:

    Context:
    {context_text}

    Question:
    {user_question}

    Answer:
    """

    prompt_content = [{"type": "text", "text": prompt_template}]

    return ChatPromptTemplate.from_messages(
        [HumanMessage(content=prompt_content)]
    )

@router.post("/query")
async def query_pdf(request: QueryRequest, weaviate_client=Depends(get_db)):
    query = request.query
    file_name = request.file

    if not query or not file_name:
        return {
            "status": "error",
            "message": "Query và file_name là bắt buộc"
        }

    # Tạo embedding cho truy vấn
    embedding_function = HuggingFaceEmbeddings()
    query_embedding = embedding_function.embed_query(query)

    # Tên class tương ứng với file
    class_name = f"Vector_Store_{file_name.replace('.', '_')}"
    class_name = re.sub(r'[^a-zA-Z0-9_]', '_', class_name)

    # Kiểm tra xem class có tồn tại không
    existing_classes = weaviate_client.collections.list_all()
    if class_name not in [collection.name for collection in existing_classes.values()]:
        return {
            "status": "error",
            "message": f"Không tìm thấy dữ liệu cho file {file_name}"
        }

    # Truy vấn Weaviate
    collection = weaviate_client.collections.get(class_name)
    result = collection.query.near_vector(
        near_vector=query_embedding,
        limit=5,
        return_properties=["summary", "original_text", "page_number"]
    )

    # Lấy danh sách các tài liệu từ kết quả
    retrieved_docs = [
        {
            "summary": obj.properties["summary"],
            "original_text": obj.properties["original_text"],
            "page_number": obj.properties.get("page_number", 1)  # Mặc định là trang 1 nếu không có
        }
        for obj in result.objects
    ]

    # Xây dựng chuỗi xử lý với LangChain
    chain_with_sources = (
        {
            "context": lambda _: retrieved_docs,  # Trả về danh sách tài liệu từ Weaviate
            "question": RunnablePassthrough(),  # Truyền câu hỏi trực tiếp
        }
        | RunnablePassthrough().assign(
            response=(
                RunnableLambda(build_prompt)
                | ChatGroq(model="llama-3.1-8b-instant", api_key=GROQ_API_KEY)  # Thay bằng Groq thay vì OpenAI
                | StrOutputParser()
            )
        )
    )

    # Thực thi chuỗi
    result = chain_with_sources.invoke(query)

    return {
        "status": "success",
        "response": result["response"],
        "context": result["context"]
    }