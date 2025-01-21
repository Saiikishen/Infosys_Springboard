from dotenv import load_dotenv

load_dotenv()

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEmbeddings

if __name__ == "__main__":

    print("Loading..")
    loader = TextLoader('./information.txt', encoding='utf-8')
    document = loader.load() 
    print(f"Loaded {len(document)} documents")

    print("Splitting Documents..")
    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200) 
    split_documents = splitter.split_documents(document)
    print(f"Split {len(document)} documents into {len(split_documents)} chunks")

    print("Started Embedding Documents...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    print("Inserting Documents")
    vector_db = PineconeVectorStore.from_documents(split_documents, embeddings, index_name="primary")
    print(f"Inserted {len(split_documents)} documents into VectorDB")
