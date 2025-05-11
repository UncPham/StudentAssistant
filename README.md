# Student Assistant: AN INTERACTIVE RAG APPLICATION


**Student Assistant** is a powerful assistant built on modern RAG (Retrieval-Augmented Generation) technology, revolutionizing the way you interact with and extract knowledge from study materials. 
The breakthrough of Student Assistant lies in its ability to transform static PDF files into an interactive and dynamic learning space. Accompanying this is an intelligent chatbot, always ready to answer any questions based on the very content of the documents.


## Features
+ **Document upload & analysis**: Upload and analyze documents  
+ **Interactive slide**: Interact directly with the content of the documents  
+ **Corrective RAG chatbot**: LLM-powered chatbot answers user questions based on the document content and online information

## Installation
To build and run Student Assistant on your computer device, follow these steps:

1. **Clone the Repository**:
```
git clone https://github.com/UncPham/StudentAssistant
```
2. **Open in Visual Studio Code**:
+ Launch Visual Studio Code.
+ Select File > Open and navigate to the cloned repository directory.
3. **Add API key**:
+ Open the *.env.example* file and add your API keys as instructed.
4. **Build the Project**:
+ Open the terminal in Visual Studio Code
+ Run the following command:
```
docker-compose up --build
```
5. **Access the Website**:
+ Your website will be available at *http://localhost:5173/*

## Usage
+ When the website is working:
+ Upload your PDF document.
+ Wait for your document to be analyzed.
+ Select the part of the document you don't understand or want to ask about.
+ Type your question.
+ The LLM will generate an answer for you.

## Acknowledgements
+ **Marker**: used for PDF document analysis tasks.
+ **Huggingface**: provides deep learning models for embedding and generating captions for images.
+ **Langchain**: used for structuring the LLM system.
+ **Vite**: used for building the frontend of the website.
