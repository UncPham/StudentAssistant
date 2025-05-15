import requests
import json
# Fix the import statement
from deepeval.metrics import GEval, AnswerRelevancyMetric, FaithfulnessMetric, ContextualRelevancyMetric
from deepeval.test_case import LLMTestCaseParams, LLMTestCase
from deepeval import assert_test, track
from time import sleep

from typing import Dict, List
import asyncio
import os
from deepeval import assert_test
import deepeval
from deepeval.dataset import Golden, EvaluationDataset
from deepeval import assert_test
import pytest

dataset = EvaluationDataset()

evaluation_params = [
    LLMTestCaseParams.INPUT,
    LLMTestCaseParams.CONTEXT,
    LLMTestCaseParams.ACTUAL_OUTPUT
]
# Backend API base URL
BASE_URL = "http://localhost:8000"

def query_pdf(query: str, file_name: str, doc_id: str = None) -> Dict:
    """Query the PDF content"""
    url = f"{BASE_URL}/chatpdf/query"
    
    payload = {
        "query": query,
        "file": file_name,
        "doc_id": doc_id
    }
    
    response = requests.post(url, json=payload)
    return response.json()


correctness_metric = GEval(
    name="Correctness",
    criteria="Determine whether the actual output is factually correct based on the expected output.",
    evaluation_steps=[
        "Check whether the facts in 'actual output' contradicts any facts in 'expected output'",
        "You should also heavily penalize omission of detail",
        "Vague language, or contradicting OPINIONS, are OK"
    ],
    evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.EXPECTED_OUTPUT],
    threshold=0.4,
)

async def evaluate_response(test_case):
    # Initialize metrics correctly
    metrics = [
        correctness_metric,
        AnswerRelevancyMetric(threshold=0.4),  
        FaithfulnessMetric(threshold=0.4),
        ContextualRelevancyMetric(threshold=0.4),  
    ]
    
    results = {}
    test = LLMTestCase(
        input=test_case["query"],
        actual_output=test_case["response"],
        expected_output=test_case["expected"],
        retrieval_context=test_case["context"]
    )

    try:
        assert_test(test, metrics)
    except AssertionError as e:
        print(f"Test failed for query: {test_case['query']}\nError: {e}")
    
    return 


async def main():
    file_name = "doc2.pdf"
    
    # Define test cases with expected outputs and contexts
    test_cases = [
        {
            "query": "Mô hình VQA nào có hiệu suất tốt nhất trong việc nhận diện tuổi theo bảng 1?",
            "expected": "Dựa trên thông tin trong Bảng 1 từ các nguồn được cung cấp: Mô hình VQA có hiệu suất tốt nhất trong việc nhận diện tuổi (Age) là Llava1.5-13B. Hiệu suất của mô hình này trên tập dữ liệu COCO khi đánh giá trên ảnh tổng hợp là: Độ chính xác (Acc.): 72.27 Điểm F1 (F1): 70.00",
            "doc_id": "/page/4/Table/7"
        },
        {
            "query": "Hình 1 trong tài liệu minh họa điều gì về OpenBias?",
            "expected": "Hình 1 làm rõ sự tương phản giữa phương pháp OpenBias, hoạt động trong kịch bản open-set (tập mở), và các phương pháp trước đây thường hoạt động trong kịch bản closed-set",
            "doc_id": "/page/0/Caption/10"
        },
        {
            "query": "Công thức (1) trong tài liệu định nghĩa điều gì?",
            "expected": "Công thức (1) định nghĩa cơ sở dữ liệu 𝐷_b của các captions và questions liên quan đến một bias cụ thể b",
            "doc_id": "/page/2/Equation/6"
        },
        {
            "query": "Công thức (6) tính điểm số severity của bias như thế nào?",
            "expected": "Công thức (6) tính điểm số severity H_b của bias bằng cách chuẩn hóa entropy của phân phối class và điều chỉnh để dễ đọc, với giá trị từ 0 (không bias) đến 1 (bias cao)",
            "doc_id": "/page/3/Equation/14"
        },
        {
            "query": "User study trong tài liệu được thực hiện như thế nào và kết quả chính là gì?",
            "expected": "User study yêu cầu người tham gia xác định hướng và mức độ bias trong ảnh, kết quả cho thấy sự đồng thuận cao giữa OpenBias và đánh giá của con người với AME=0.15 và agreement 67% về majority class.",
            "doc_id": "/page/5/TextInlineMath/2"
        },
        {
            "query": "Tại sao việc nghiên cứu bias trong các mô hình sinh ảnh lại quan trọng?",
            "expected": "Nghiên cứu bias trong các mô hình sinh ảnh quan trọng để đảm bảo fairness, tránh perpetuating stereotypes, và nâng cao tính inclusive của AI.",
            "doc_id": "/page/0/Text/7"
        },
        {
            "query": "OpenBias đánh giá bias trong các mô hình sinh ảnh như thế nào?",
            "expected": "OpenBias đánh giá bias bằng cách sử dụng VQA để nhận diện class của bias trong ảnh sinh ra và tính toán phân phối class để xác định mức độ bias.",
            "doc_id": "/page/2/TextInlineMath/15"
        },
        {
            "query": "Sự khác biệt giữa context-aware bias và context-free bias là gì?",
            "expected": "Context-aware bias xem xét bias trong ngữ cảnh cụ thể của từng caption, trong khi context-free bias xem xét bias tổng thể trên toàn bộ tập dữ liệu mà không phụ thuộc vào ngữ cảnh caption.",
            "doc_id": "/page/3/TextInlineMath/4"
        },
        {
            "query": "Mô hình nào có KL divergence thấp nhất cho bias về giới tính trên tập dữ liệu Flickr 30k?",
            "expected": "SD-XL có KL divergence thấp nhất cho bias về giới tính trên Flickr 30k với giá trị 0.006.",
            "doc_id": "/page/4/Table/9"
        },
        {
            "query": "Công thức (4) được sử dụng để tính toán gì trong pipeline OpenBias?",
            "expected": "Công thức (4) được sử dụng trong pipeline OpenBias để tính toán xác suất (probability) của một lớp cụ thể (c) thuộc về tập hợp các lớp khả thi (Cb) cho một thiên kiến (b), dựa trên hình ảnh được tạo ra từ một chú thích (caption) cụ thể (t)",
            "doc_id": "/page/3/Equation/5"
        },
        {
            "query": "Một số bias mới được phát hiện bởi OpenBias là gì?",
            "expected": "Một số bias mới bao gồm laptop brand, train color, horse breed, child gender, child race, và person attire",
            "doc_id": "/page/5/Caption/7"        
        },
        {
            "query": "Tại sao việc phát hiện bias trong tập mở lại quan trọng?",
            "expected": "việc phát hiện thiên kiến trong tập mở là quan trọng vì nó vượt qua hạn chế của các phương pháp tập đóng, cho phép khám phá và định lượng các thiên kiến mới, đặc thù theo ngữ cảnh và chưa từng được nghiên cứu, điều này là cần thiết để hiểu đầy đủ và giải quyết các vấn đề công bằng và an toàn trong các mô hình T2I đang ngày càng phổ biến",
            "doc_id": "/page/0/Text/7"
        },
        {
            "query": "Giải thích công thức  (5)",
            "expected": "Công thức (5) trong pipeline OpenBias được sử dụng để tính toán xác suất trung bình (average probability) của một lớp cụ thể (c) thuộc về tập hợp các lớp khả thi (Cb) cho một thiên kiến (b), trên toàn bộ tập hợp các chú thích và câu hỏi liên quan đến thiên kiến đó (Db).",
            "doc_id": "/page/3/Equation/9"
        },
        {
            "query": "Ngoài Stable Diffusion, tài liệu có nghiên cứu các mô hình sinh ảnh khác không?",
            "expected": "Dựa trên thông tin trong tài liệu, nghiên cứu này chủ yếu tập trung vào việc nghiên cứu và đánh giá thiên kiến trong các mô hình sinh ảnh Stable Diffusion",
            "doc_id": "/page/0/Text/5"
        },
        {
            "query": "Hạn chế của OpenBias là gì?",
            "expected": "Tài liệu chủ yếu tập trung vào việc nghiên cứu và đánh giá thiên kiến trong các phiên bản của mô hình Stable Diffusion Tuy nhiên, tài liệu cũng nêu rõ rằng pipeline OpenBias là một kiến trúc mô-đun và linh hoạt.... Nó được thiết kế để có thể dễ dàng mở rộng (easily extended) để nghiên cứu thiên kiến trong các loại mô hình khác.",
            "doc_id": "/page/7/Text/6"
        },
        {
            "query": "OpenBias có thể được sử dụng cho các ngôn ngữ khác ngoài tiếng Anh không?",
            "expected": "mặc dù các nguồn không trực tiếp nói rằng OpenBias hỗ trợ các ngôn ngữ khác ngoài tiếng Anh, nhưng tính mô-đun của nó cho thấy khả năng nó có thể được mở rộng hoặc điều chỉnh để làm việc với các ngôn ngữ khác.",
            "doc_id": None
        },
        {
            "query": "OpenBias đã được ứng dụng trong thực tế chưa?",
            "expected": "OpenBias được trình bày như một công cụ nghiên cứu và một khuôn khổ mới để phát hiện thiên kiến một cách linh hoạt..., và các tài liệu tập trung vào việc thử nghiệm và chứng minh khả năng của nó trong môi trường nghiên cứu và thử nghiệm..., chứ không phải việc đã được ứng dụng rộng rãi trong thực tế.",
            "doc_id": None        
        },
        {
            "query": "Giải thích OpenBias pipeline thể hiện trong hình 2",
            "expected": "Hình 2 thể hiện một quy trình tự động ba bước, từ việc sử dụng LLM để đề xuất các thiên kiến tiềm năng từ chú thích, đến việc sử dụng mô hình sinh ảnh T2I để tạo hình ảnh, và cuối cùng là sử dụng VQA để đánh giá và định lượng mức độ của các thiên kiến đó trong hình ảnh được tạo ra.... Kiến trúc này mang tính mô-đun, cho phép thay thế dễ dàng các thành phần (LLM, Mô hình Sinh ảnh, VQA) bằng các phiên bản khác hoặc đặc thù hơn",
            "doc_id": "/page/2/Caption/1"
        },
        {
            "query": "OpenBias có thể được mở rộng để nghiên cứu bias trong các tập dữ liệu multimodal thực tế như thế nào?",
            "expected": "khả năng mô-đun và linh hoạt của pipeline OpenBias... cho phép thay thế thành phần sinh ảnh bằng nguồn dữ liệu hình ảnh thực tế của tập dữ liệu multimodal. Điều này cho phép OpenBias trở thành một công cụ để nghiên cứu các thiên kiến không chỉ trong các mô hình sinh ảnh T2I mà còn tiềm năng trong chính các tập dữ liệu multimodal thực tế được sử dụng để huấn luyện hoặc đánh giá mô hình. Việc sử dụng các tập dữ liệu như Flickr 30k và COCO (vốn là các tập dữ liệu multimodal thực tế chứa hình ảnh và chú thích) trong các thí nghiệm... cũng cho thấy khả năng tương thích của OpenBias với loại dữ liệu này.",
            "doc_id": "/page/3/TextInlineMath/13"
        },
        {
            "query": "OpenBias có thể được dùng để phân tích bias trong các mô hình ngôn ngữ không?",
            "expected": "OpenBias sử dụng LLM như một công cụ để phát hiện thiên kiến trong các mô hình khác (T2I) hoặc tập dữ liệu, nhưng không được mô tả là có thể phân tích thiên kiến bên trong chính LLM",
            "doc_id": None
        },
    ]
    
    print("\nStarting evaluation...")
    print("=" * 50)
    
    for test_case in test_cases:
        print(f"\nProcessing query: {test_case['query']}")
        
        # Get API response
        response = query_pdf(test_case["query"], file_name, doc_id=test_case["doc_id"])
        test_case["response"] = response.get("answer", "")
        rag_documents = response["context"]["rag_documents"]

        context = []

        # Lặp qua rag_documents để lấy nội dung tài liệu
        for doc in rag_documents:
            document_content = doc.get("document", "")
            if document_content:
                context.append(document_content)

        test_case["context"] = context
        
        sleep(60)
        # Evaluate using multiple metrics
        evaluation = await evaluate_response(test_case)

        sleep(60)

@pytest.mark.asyncio
def test_rag_evaluation():
    """Pytest-compatible test function for deepeval discovery."""
    asyncio.run(main())

if __name__ == "__main__":
    test_rag_evaluation()