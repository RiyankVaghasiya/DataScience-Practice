from langchain_huggingface import HuggingFaceEmbeddings

embedding = HuggingFaceEmbeddings(model = 'sentence-transformers/all-MiniLM-L6-v2')

# text = "delhi is the capital of india"

#we can also send multiple text or documents
documents = [
    "Delhi is the capital of India",
    "Kolkata is the capital of West Bengal",
    "Paris is the capital of France"
]


vector = embedding.embed_documents(documents)

print(str(vector))