import os
import sys
import glob
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Updated imports
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_chroma.vectorstores import Chroma
from langchain_openai.llms import OpenAI
from langchain.chains import RetrievalQA

# Updated document loaders
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

def main():
    # Load OpenAI API key
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        print("Please set your OPENAI_API_KEY in the .env file.")
        sys.exit(1)
    
    # Define the folder path (change 'data' to your folder name)
    folder_path = './data'
    if not os.path.exists(folder_path):
        print(f"Folder '{folder_path}' does not exist.")
        sys.exit(1)
    
    # Read all files in the folder
    documents = []
    for filepath in glob.glob(os.path.join(folder_path, '**/*.*'), recursive=True):
        if os.path.isfile(filepath):
            ext = os.path.splitext(filepath)[1].lower()
            try:
                if ext == '.txt':
                    loader = TextLoader(filepath, encoding='utf-8')
                    documents.extend(loader.load_and_split())
                elif ext == '.pdf':
                    loader = PyPDFLoader(filepath)
                    documents.extend(loader.load_and_split())
                else:
                    print(f"Unsupported file format: {filepath}")
            except Exception as e:
                print(f"Error reading '{filepath}': {e}")
    
    if not documents:
        print("No documents found in the folder.")
        sys.exit(1)
    
    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(documents)
    
    # Initialize embeddings and vector store
    embeddings = OpenAIEmbeddings()
    vector_store = Chroma(embedding_function=embeddings, persist_directory="./chroma_store")
    
    # Add texts to vector store in batches
    batch_size = 500  # Adjust this number as needed
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        vector_store.add_documents(batch_texts)
    
    # No need to call persist() anymore
    # vector_store.persist()  # Remove or comment out this line
    
    # Set up retriever
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    
    # Set up the language model
    llm = OpenAI(temperature=0.7)
    
    # Create the RetrievalQA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",  # Options: 'stuff', 'map_reduce', 'refine', 'map_rerank'
        retriever=retriever
    )
    
    # Interactive prompt for user queries
    print("The system is ready. You can now ask questions about the content.")
    while True:
        query = input("Enter your question (or type 'exit' to quit): ")
        if query.lower() in ('exit', 'quit'):
            break
        try:
            response = qa_chain.run(query)
            print(f"\nAnswer: {response}\n")
        except Exception as e:
            print(f"An error occurred: {e}\n")
            
if __name__ == "__main__":
    main()
