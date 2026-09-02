from langchain_community.document_loaders import Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore

# 1. Load the Word document
loader = Docx2txtLoader("data/End-to-End Machine Learning and MLOps.docx")
documents = loader.load()
print(f"Loaded {len(documents)} document(s)")

# 2. Cut it into small chunks
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,      # each chunk is ~500 characters
    chunk_overlap=50     # slices overlap a little so we don't lose meaning
)
chunks = splitter.split_documents(documents)
print(f"Split into {len(chunks)} chunks")

# 3. Turn chunks into number fingerprints (embeddings) and store in Qdrant
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

vectorstore = QdrantVectorStore.from_documents(
    chunks,
    embeddings,
    path="./qdrant_data",
    collection_name="mlops_notes",
)

print("Done! Your documents are now stored in the vector database.")