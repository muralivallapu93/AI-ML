import os
from dotenv import load_dotenv
from pydantic.v1 import BaseModel, Field
import numpy as np
import time
import json
import openai
from llama_index.core import (
    SimpleDirectoryReader, 
    load_index_from_storage, 
    VectorStoreIndex, 
    StorageContext
)
from llama_index.llms.azure_openai import AzureOpenAI
from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core import Settings
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.core.storage.chat_store import SimpleChatStore
import faiss
from pathlib import Path

# Dimensions of text-ada-embedding-002 = 1536
d = 1536
faiss_index = faiss.IndexFlatL2(d)

# Load environment variables
current_directory = os.getcwd()
BASE_DIR = Path(__file__).resolve().parent
env_path = os.path.join(current_directory, '.env')
load_dotenv(dotenv_path=env_path)

AZURE_OPENAI_API_KEY = os.getenv('AZURE_OPENAI_API_KEY')
AZURE_OPENAI_API_ENDPOINT = os.getenv('AZURE_OPENAI_API_ENDPOINT')
AZURE_OPENAI_API_VERSION = os.getenv('AZURE_OPENAI_API_VERSION')
AZURE_OPENAI_DEPLOYMENT_NAME = os.getenv('AZURE_OPENAI_DEPLOYMENT_NAME')
AZURE_OPENAI_MODEL_NAME = os.getenv('AZURE_OPENAI_MODEL_NAME')

# Setting up Azure OpenAI and embedding models
api_key = AZURE_OPENAI_API_KEY
azure_endpoint = AZURE_OPENAI_API_ENDPOINT
api_version = AZURE_OPENAI_API_VERSION

llm = AzureOpenAI(
    model="gpt-35-turbo",
    deployment_name=AZURE_OPENAI_DEPLOYMENT_NAME,
    api_key=api_key,
    azure_endpoint=azure_endpoint,
    api_version=api_version,
)

embed_model = AzureOpenAIEmbedding(
    model="text-embedding-ada-002",
    deployment_name=AZURE_OPENAI_DEPLOYMENT_NAME,
    api_key=api_key,
    azure_endpoint=azure_endpoint,
    api_version=api_version,
)

Settings.embed_model = embed_model
Settings.llm = llm

chat_store = SimpleChatStore()

# Example prompt
prompt = (
    "Evaluate the SDD document against the template provided, ensuring it adheres to the organizational standards of Solution Design. "
    "For each of the criteria, provide a 250-300 word object with: "
    "1. Question: (context_str)"
    "2. Compliance (Yes/No): Does the SDD meet this requirement?"
    "3. Rationale: Briefly justify your answer, referencing the SDD."
    "4. Action: Suggest Improvements if non-compliant."
    "5. Impact: Rate the severity of non-compliance (1-5)."
    "Finally, conclude with:"
    "Approval Status: Calculate the overall compliance percentage and determine the approval status using these thresholds: 'Approved', 'Pending', or 'Rejected'."
    "Accuracy Score: Accuracy of data provided in document based on the information coverage of the context."
    "Compliance score: Percentage based on the weighted importance of each criterion."
    "Feedback: Conclude this document based on the review of all criteria."
)

class GovernancePOC:
    def __init__(self, llm, embed_model, settings):
        self.settings = settings
        self.llm = llm

    def create_vector_store(self):
        self.settings.embed_model = embed_model
        doc_path = os.path.join(BASE_DIR, 'static', 'governance_folder')
        documents = SimpleDirectoryReader(input_dir=doc_path).load()
        vector_store = FaissVectorStore(faiss_index=faiss_index)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        index = VectorStoreIndex.from_documents(documents, storage_context=storage_context)
        index.storage_context.persist(persist_dir=os.path.join(BASE_DIR, 'VectorStorage'))

    def load_vector_db(self, db_directory=None):
        if db_directory is None:
            persist_dir = os.path.join(BASE_DIR, 'VectorStorage')
        else:
            persist_dir = db_directory
        vector_store = FaissVectorStore.from_persist_dir(persist_dir)
        storage_context = StorageContext.from_defaults(vector_store=vector_store, persist_dir=persist_dir)
        index = load_index_from_storage(storage_context=storage_context)
        return index

    def gen_ai_chat_engine(self, index, query, prompt, user_id="abc123"):
        memory = ChatMemoryBuffer.from_defaults(token_limit=3500, chat_store=chat_store, chat_store_key=user_id)
        chat_engine = index.as_chat_engine(
            chat_mode="condense_plus_context",
            memory=memory,
            llm=llm,
            context_prompt=prompt,
            verbose=False,
        )
        response = chat_engine.chat(query)
        chat_store.persist(persist_path="chat_store.json")
        return response.response

    def validate_question(self, question, indexes):
        results = self.gen_ai_chat_engine(indexes, question, prompt)
        return results

    def validate_all_questions(self, questions, indexes):
        validation_results = [(question, self.validate_question(question, indexes)) for question in questions]
        return validation_results

def main(sdd_file_name):
    user_uploaded_doc = os.path.join(BASE_DIR, sdd_file_name)
    sdd_docs = SimpleDirectoryReader(input_files=[user_uploaded_doc]).load_data()
    user_uploaded_index = VectorStoreIndex.from_documents(sdd_docs)
    
    prompt2 = (
        "Extract and generate 20 advanced validation questions from the governance documents to assess the SDD. Each question should:"
        "1. Criterion: Identify the specific governance criterion or policy."
        "2. Question: Formulate a detailed question that probes compliance with the identified criterion."
    )
    
    obj = GovernancePOC(llm, embed_model, Settings)
    obj.create_vector_store()
    
    questions = [
        "Are the project requirements clearly stated in the SDD document?",
        "Are the performance requirements clearly defined in the SDD document?"
    ]
    results = obj.validate_all_questions(questions, user_uploaded_index)
    print(results[0][1])
    
    index1 = obj.load_vector_db()
    query = "generate the top 20 questions that are required to validate the Solution Design Document, on the basis of fields and format provided"
    questions1 = obj.gen_ai_chat_engine(index1, query, prompt=prompt2, user_id="abc1234")
    
    results1 = obj.validate_all_questions([questions1], user_uploaded_index)
    print(results1)
    
    with open("savedata.json", "w") as save_file:
        json.dump(results1, save_file, indent=6)
    
    sdd_file_name = "static/files/PG-TestCase1.docx"
    main(sdd_file_name)

if __name__ == "__main__":
    main("static/files/TestCase1.docx")
