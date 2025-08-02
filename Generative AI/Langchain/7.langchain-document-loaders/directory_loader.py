from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader

loader = DirectoryLoader(
    path='books',
    glob='*.pdf',
    loader_cls=PyPDFLoader
)


docs = loader.load() # here we can use lazy_load instead of load function to save time

print(docs[1].page_content)

