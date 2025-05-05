from fastapi import APIRouter, Request, UploadFile, File, Depends, Body 
import os
from marker.config.parser import ConfigParser
from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict
from marker.output import text_from_rendered
from backend.config import CACHE_DIR
# RAG
from marker.config.printer import CustomClickPrinter
from marker.logger import configure_logging
from marker.output import save_output

from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
import uuid
from langchain.vectorstores import Chroma
from langchain.storage import InMemoryStore
from langchain.schema.document import Document
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings  # Updated import
from langgraph.graph import StateGraph, START, END
from tavily import TavilyClient
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
from typing import List, Dict
from typing_extensions import TypedDict
import json
from bs4 import BeautifulSoup
import groq  # Add this import for handling RateLimitError

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from backend.config import GROQ_API_KEY, TAVILY_API_KEY, GOOGLE_API_KEY, CACHE_DIR
from transformers import AutoModelForCausalLM, AutoTokenizer
import google.generativeai as genai

# import logging

from pydantic import BaseModel
import time

class QueryRequest(BaseModel):
    query: str
    file: str
    doc_id: str | None = None

class PolygonRequest(BaseModel):
    file: str

# Định nghĩa GraphState
class GraphState(TypedDict):
    question: str
    generation: str
    search: str
    documents: List[Document]
    steps: List[str]

router = APIRouter()

# Khởi tạo các thành phần của Corrective RAG
embedding_function = HuggingFaceEmbeddings(
    model_name="intfloat/multilingual-e5-large-instruct",
    cache_folder=CACHE_DIR,
    model_kwargs={'device': 'cpu'}
)

llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-001", temperature=0.2, api_key=GOOGLE_API_KEY)

# Prompt cho RAG
prompt = PromptTemplate(
    template="""
    [ROLE]
    You are an expert technical consultant.

    [TASK]
    Your task is to answer questions accurately based solely on the provided documents.

    [MANDATORY RULES TO FOLLOW IN TASK COMPLETION]
    - Use only documents directly related to the question to construct the answer.
    - Provide a precise, structured, and accurate answer without repeating the question.
    - Do not fabricate facts or include information not present in the provided documents.
    - If the documents do not contain enough information to answer the question or the answer is unclear, explicitly state: "The provided documents do not contain enough information to answer the question," and explain why you believe this to be the case.

    [TASK COMPLETION STEPS]
    Strictly follow each of these steps in order to complete the task:
    1. Detect the question's source language. Your answer must be in the same language as the question.
    2. Extract key information relevant to the question from the related documents.
    3. Synthesize information from the relevant documents.
    4. Based on the synthesized information, provide the final answer.

    [CONTEXT DOCUMENT TO PROVIDE CONTEXT FOR QUESTION]
    Documents: {documents}

    [QUESTION TO ANSWER]
    Question: {question}

    [YOUR ANSWER]
    Answer:
    """,
    input_variables=["question", "documents"],
)

# Prompt cho retrieval grader
retrieval_prompt = PromptTemplate(
    template="""You are a grader evaluating the relevance of a document to a question in a quiz. You will be given:
    1/ a QUESTION
    2/ a FACT, which is a text excerpt from a document.

    Return ONLY a JSON object with a single "score" key and value either "yes" or "no".

    You are grading RELEVANCE RECALL:
    A score of 1 means that ANY of the statements in the FACT are relevant to the QUESTION. 
    A score of 0 means that NONE of the statements in the FACT are relevant to the QUESTION. 
    1 is the highest (best) score. 0 is the lowest score you can give. 
    
    Avoid simply stating the correct answer at the outset.
    
    Question: {question} \n
    Fact: \n\n {documents} \n\n

    Assign a binary score:
    - 'yes' if the FACT has any relevance to the QUESTION, even if minimal.
    - 'no' if the FACT is completely unrelated and provides no useful information.

    Example response: {{"score": "yes"}} or {{"score": "no"}}
    """,
    input_variables=["question", "documents"],
)

# Khởi tạo rag_chain và retrieval_grader
rag_chain = prompt | llm | StrOutputParser()
retrieval_grader = retrieval_prompt | llm | JsonOutputParser()

# Khởi tạo web_search_tool
# os.environ["TAVILY_API_KEY"] = TAVILY_API_KEY
web_search_tool = TavilyClient(api_key=TAVILY_API_KEY)

# Định nghĩa các hàm của StateGraph
def retrieve(state):
    print("Retrieving documents...")
    question = state["question"]
    documents = state.get("documents", [])  # Sử dụng documents từ state
    steps = state["steps"]
    steps.append("retrieve_documents")
    return {"documents": documents, "question": question, "steps": steps}

def generate(state):
    print("Generating answer...")
    question = state["question"]
    documents = state["documents"]
    # Print the formatted prompt
    formatted_prompt = prompt.format(
        documents="\n".join([doc.page_content for doc in documents]),
        question=question
    )
    print("Generated Prompt:")
    print(formatted_prompt)
    generation = rag_chain.invoke({"documents": [doc.page_content for doc in documents], "question": question})
    print(f"Generation: {generation}")
    steps = state["steps"]
    steps.append("generate_answer")
    return {
        "documents": documents,
        "question": question,
        "generation": generation,
        "steps": steps,
    }

def grade_documents(state):
    print("Grading documents...")
    question = state["question"]
    documents = state["documents"]
    steps = state["steps"]
    steps.append("grade_document_retrieval")
    filtered_docs = []
    search = "No"
    for d in documents:
        score = retrieval_grader.invoke(
            {"question": question, "documents": d.page_content}
        )
        grade = score["score"]
        if grade == "yes":
            filtered_docs.append(d)
        else:
            search = "Yes"
            continue

    return {
        "documents": filtered_docs,
        "question": question,
        "search": search,
        "steps": steps,
    }

async def web_search(state):
    print("Performing web search...")
    question = state["question"]
    documents = state.get("documents", [])
    steps = state["steps"]
    steps.append("web_search")
    try:
        search_results = web_search_tool.search(
            query=question,
            search_depth="advanced",
            max_results=5
        )

        print(search_results)
        
        # Convert search results to Document objects
        for result in search_results["results"]:
            print(f"Web search result: {result}")
            documents.append(
                Document(
                    page_content=result.get("content", ""),
                    metadata={
                        "url": result.get("url", ""),
                        "source": "web_search"
                    }
                )
            )
        print(f"Found {len(search_results)} web search results")
        
    except Exception as e:
        print(f"Web search error: {str(e)}")
        # Continue with existing documents if search fails

def decide_to_generate(state):
    search = state["search"]
    return "search" if search == "Yes" else "generate"

# Xây dựng StateGraph
workflow = StateGraph(GraphState)
workflow.add_node("retrieve", retrieve)
workflow.add_node("grade_documents", grade_documents)
workflow.add_node("generate", generate)
workflow.add_node("web_search", web_search)
workflow.add_edge(START, "retrieve")
workflow.add_edge("retrieve", "grade_documents")
workflow.add_conditional_edges(
    "grade_documents",
    decide_to_generate,
    {
        "search": "web_search",
        "generate": "generate",
    },
)
workflow.add_edge("web_search", "generate")
workflow.add_edge("generate", END)
custom_graph = workflow.compile()

def get_db(request: Request):
    return request.app.weaviate_client

# Hàm thiết lập schema Weaviate
def setup_weaviate_schema(client: weaviate.WeaviateClient, class_name: str):
    schema = {
        "class": class_name,
        "properties": [
            {"name": "doc_id", "dataType": ["string"]},
            {"name": "original_text", "dataType": ["text"]},
            {"name": "page_number", "dataType": ["int"]},
            {"name": "polygon", "dataType": ["text"]},
            {"name": "bbox", "dataType": ["text"]},
            {"name": "html", "dataType": ["text"]}
        ],
        "vectorizer": "none"
    }
    existing_classes = client.collections.list_all()
    if class_name not in [collection.name for collection in existing_classes.values()]:
        client.collections.create_from_dict(schema)

# Hàm làm sạch HTML và trích xuất văn bản
def clean_html(html_text: str) -> str:
    soup = BeautifulSoup(html_text, "html.parser")
    for tag in soup.find_all(['sup', 'i', 'b']):
        tag.unwrap()
    text = soup.get_text(separator=" ", strip=True)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Hàm chia văn bản thành các chunk
def chunk_text(texts: List[str], max_chars: int = 500, overlap: int = 100) -> List[str]:
    # Kiểm tra overlap hợp lệ
    if overlap >= max_chars:
        raise ValueError("Overlap phải nhỏ hơn max_chars")
    if overlap < 0:
        raise ValueError("Overlap không được âm")

    # Gộp tất cả các text thành một chuỗi lớn
    full_text = " ".join(texts)
    
    # Khởi tạo danh sách các chunk
    chunks = []
    start = 0
    
    # Chia chuỗi lớn thành các chunk
    while start < len(full_text):
        # Xác định vị trí kết thúc chunk (tối đa max_chars)
        end = start + max_chars
        # Nếu đã đến cuối chuỗi, thêm phần còn lại vào chunks và thoát
        if end >= len(full_text):
            chunks.append(full_text[start:].strip())
            break
        # Tìm khoảng trắng gần nhất trước vị trí end để không cắt ngang từ
        while end > start and full_text[end] != " ":
            end -= 1
        # Nếu không tìm thấy khoảng trắng, cắt đúng tại max_chars
        if end == start:
            end = start + max_chars
        # Thêm chunk vào danh sách
        chunks.append(full_text[start:end].strip())
        # Cập nhật vị trí bắt đầu cho chunk tiếp theo, bao gồm overlap
        start = end - overlap
        # Đảm bảo start không âm
        if start < 0:
            start = 0
    
    return chunks

# Hàm trích xuất nội dung từ JSON
def extract_content_from_json(json_data: Dict) -> List[Dict]:
    extracted_data = []
    
    for page in json_data.get("children", []):
        if page["block_type"] != "Page":
            continue
        
        page_number = int(page["id"].split("/")[2])
        for child in page.get("children", []):
            block_type = child["block_type"]
            block_id = child["id"]
            html_content = child.get("html", "")
            polygon = json.dumps(child.get("polygon", []))  # Chuyển thành JSON string
            bbox = json.dumps(child.get("bbox", []))        # Chuyển thành JSON string
            
            if block_type in ["Text", "TextInlineMath"]:
                cleaned_text = clean_html(html_content)
                if not cleaned_text:
                    continue

                extracted_data.append({
                    "doc_id": block_id,
                    "original_text": cleaned_text,
                    "html": cleaned_text,  # Lưu HTML đã làm sạch
                    "page_number": page_number,
                    "polygon": polygon,
                    "bbox": bbox
                })
            
            elif block_type == "ListGroup":
                for list_item in child.get("children", []):
                    if list_item["block_type"] != "ListItem":
                        continue
                    cleaned_text = clean_html(list_item.get("html", ""))
                    if not cleaned_text:
                        continue
                    
                    extracted_data.append({
                        "doc_id": list_item["id"],
                        "original_text": cleaned_text,
                        "html": cleaned_text,
                        "page_number": page_number,
                        "polygon": json.dumps(list_item.get("polygon", [])),
                        "bbox": json.dumps(list_item.get("bbox", []))
                    })
            elif block_type == "FigureGroup":
                for list_item in child.get("children", []):
                    if list_item["block_type"] != "Caption":
                        continue
                    cleaned_text = clean_html(list_item.get("html", ""))
                    if not cleaned_text:
                        continue
                    
                    extracted_data.append({
                        "doc_id": list_item["id"],
                        "original_text": cleaned_text,
                        "html": cleaned_text,
                        "page_number": page_number,
                        "polygon": json.dumps(list_item.get("polygon", [])),
                        "bbox": json.dumps(list_item.get("bbox", []))
                    })

            elif block_type == "Equation":
                cleaned_text = html_content
                if not cleaned_text:
                    continue

                extracted_data.append({
                    "doc_id": block_id,
                    "original_text": cleaned_text,
                    "html": cleaned_text,  # Lưu HTML đã làm sạch
                    "page_number": page_number,
                    "polygon": polygon,
                    "bbox": bbox
                })

            elif block_type == "TableGroup":
                for list_item in child.get("children", []):
                    if list_item["block_type"] != "Table":
                        continue
                    cleaned_text = list_item.get("html", "")
                    if not cleaned_text:
                        continue
                    
                    extracted_data.append({
                        "doc_id": list_item["id"],
                        "original_text": cleaned_text,
                        "html": cleaned_text,
                        "page_number": page_number,
                        "polygon": json.dumps(list_item.get("polygon", [])),
                        "bbox": json.dumps(list_item.get("bbox", []))
                    })

    
    return extracted_data

@router.post("/embeddings")
async def embeddings_pdf(file: UploadFile = File(...), weaviate_client=Depends(get_db)):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        shutil.copyfileobj(file.file, temp_file)
        temp_file_path = temp_file.name 

    config = {
        "output_format": "json",
        # "cache_dir": "models_cache",  # Thêm đường dẫn cache
        # "model_cache": True,  # Bật cache cho mô hình
        "ADDITIONAL_KEY": "VALUE"
    }
    config_parser = ConfigParser(config)

    converter = PdfConverter(
        config=config_parser.generate_config_dict(),
        artifact_dict=create_model_dict(),
        processor_list=config_parser.get_processors(),
        renderer=config_parser.get_renderer(),
        llm_service=config_parser.get_llm_service()
    )
    rendered = converter(temp_file_path)
    
    text, _, images = text_from_rendered(rendered)

    try:
        json_obj = json.loads(text)
    except json.JSONDecodeError:
        os.remove(temp_file_path)
        return {
            "status": "error",
            "message": "Invalid JSON format in rendered text",
        }

    extracted_data = extract_content_from_json(json_obj)
    texts = [item["original_text"] for item in extracted_data]

    chunked_texts = chunk_text(texts)

    # Kiểm tra xem texts có rỗng không
    if not chunked_texts:
        os.remove(temp_file_path)
        return {
            "status": "error",
            "message": "Không tìm thấy nội dung văn bản trong PDF để xử lý",
        }
    else:
        print("Found chunked_texts: ", len(chunked_texts))

    embeddings = embedding_function.embed_documents(chunked_texts)
    
    class_name = f"Vector_Store_{file.filename.replace('.', '_')}"
    class_name = re.sub(r'[^a-zA-Z0-9_]', '_', class_name)
    setup_weaviate_schema(weaviate_client, class_name)

    polygon_collection = weaviate_client.collections.get(f"{class_name}_Polygon")
    rag_collection = weaviate_client.collections.get(f"{class_name}_RAG")

    for item in extracted_data:
        doc_data = {
            "doc_id": item["doc_id"],
            "original_text": item["original_text"],
            "page_number": item["page_number"],
            "polygon": item["polygon"],
            "bbox": item["bbox"],
            "html": item["html"]
        }
        polygon_collection.data.insert(
            properties=doc_data
        )

    for item, embedding in zip(chunked_texts, embeddings):
        doc_data = {
            "original_text": item,
        }
        rag_collection.data.insert(
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
            [f"[{i+1}]: {doc['original_text']}\n" for i, doc in enumerate(docs)]
        )
    else:
        context_text = "No context available."

    # Tạo template prompt
    prompt_template = f"""
    You are a technical expert tasked with answering questions accurately based solely on the provided documents.
    Your role is to:
    - Evaluate each document to determine its relevance to the question.
    - Use only documents directly related to the question to formulate your answer.
    - Ignore any documents that are irrelevant to the question.
    - If multiple documents contain relevant information, synthesize and summarize the information concisely.
    - Cite each piece of information using the document number in square brackets, e.g., [3], referring only to the number in brackets after the document.
    - Provide a precise, structured, and accurate answer without repeating the question.
    - Do not fabricate facts or include information not present in the provided documents.
    - If the documents do not contain enough information to answer the question or the answer is unclear, explicitly state: "The provided documents do not contain enough information to answer the question."
    - Avoid using brackets for anything other than document references.

    Context:
    {context_text}

    {user_question}

    Answer:
    """

    prompt_content = [{"type": "text", "text": prompt_template}]

    return ChatPromptTemplate.from_messages(
        [HumanMessage(content=prompt_content)]
    )

@router.post("/query")
async def query_pdf(request: QueryRequest, weaviate_client=Depends(get_db)):
    # Log the incoming request data
    print(f"Received query: {request.query}, file: {request.file}, doc_id: {request.doc_id}")

    query = request.query
    file_name = request.file
    doc_id = request.doc_id

    if not query or not file_name:
        return {
            "status": "error",
            "message": "Query và file_name là bắt buộc"
        }

    # Tên class tương ứng với file
    class_name = f"Vector_Store_{file_name.replace('.', '_')}_RAG"
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

    # Tạo embedding cho truy vấn
    query_embedding = embedding_function.embed_query(query)
    
    # Query for the nearest vector match, excluding the exact doc_id
    result = collection.query.near_vector(
        near_vector=query_embedding,
        limit=5,
        return_properties=["original_text"]
    )

    # if doc_id:
    #     try:
    #         doc_id_result = collection.query.fetch_objects(
    #             filters=weaviate.classes.query.Filter.by_property("doc_id").equal(doc_id),
    #             return_properties=["original_text"],
    #             limit=1
    #         )
    #         if doc_id_result.objects:
    #             original_text = doc_id_result.objects[0].properties.get("original_text", "")
    #             query = f"The main context: {original_text} \n Answer the question: {query}"
    #             print(f"Updated query with original_text: {query}")
    #         else:
    #             print(f"No document found for doc_id: {doc_id}")
    #     except weaviate.exceptions.WeaviateQueryError as e:
    #         print(f"Warning: Failed to fetch document for doc_id {doc_id}: {str(e)}")

    # Lấy danh sách các tài liệu từ kết quả
    retrieved_docs =  [
        Document(
            page_content=obj.properties["original_text"],
            metadata={}
        )
        for obj in result.objects
    ]

    # # Xây dựng chuỗi xử lý với LangChain
    # chain_with_sources = (
    #     {
    #         "context": lambda _: retrieved_docs,  # Trả về danh sách tài liệu từ Weaviate
    #         "question": RunnablePassthrough(),  # Truyền câu hỏi trực tiếp
    #     }
    #     | RunnablePassthrough().assign(
    #         response=(
    #             RunnableLambda(build_prompt)
    #             | ChatGoogleGenerativeAI(model="gemini-2.0-flash-001", temperature=0.2, api_key=GOOGLE_API_KEY)  # Thay bằng Groq thay vì OpenAI
    #             | StrOutputParser()
    #         )
    #     )
    # )

    # Thực thi chuỗi
    # result = chain_with_sources.invoke(query)

     # Gọi custom_graph để xử lý
    state = {
        "question": query,
        "documents": retrieved_docs,
        "steps": [],
        "generation": "",
        "search": "No"
    }
    result = await custom_graph.ainvoke(state)

    return {
        "status": "success",
        "response": result["generation"],
        "context": [
            {"original_text": doc.page_content, "doc_id": doc.metadata}
            for doc in result["documents"]
        ]
    }

@router.post("/polygon")
async def query_pdf(request: PolygonRequest, weaviate_client=Depends(get_db)):
    file_name = request.file

    # Tạo tên class tương ứng
    class_name = f"Vector_Store_{file_name.replace('.', '_')}_Polygon"
    class_name = re.sub(r'[^a-zA-Z0-9_]', '_', class_name)

    # Kiểm tra class tồn tại
    existing_classes = weaviate_client.collections.list_all()
    if class_name not in [collection.name for collection in existing_classes.values()]:
        return {
            "status": "error",
            "message": f"Không tìm thấy dữ liệu cho file {file_name}"
        }

    # Truy vấn toàn bộ id và polygon
    collection = weaviate_client.collections.get(class_name)

    objects = list(collection.iterator(return_properties=["doc_id", "polygon"]))
    # Chuẩn hóa dữ liệu trả về
    formatted_data = [
        {
            "doc_id": obj.properties["doc_id"],
            "polygon": json.loads(obj.properties["polygon"])  # Parse chuỗi JSON thành mảng
        }
        for obj in objects
        if "doc_id" in obj.properties and "polygon" in obj.properties
    ]
    
    return {
        "status": "success",
        "data": formatted_data
    }
