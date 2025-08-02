from langchain_community.document_loaders import WebBaseLoader
from langchain_community.document_loaders import TextLoader
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

model = ChatGroq(
    model="llama3-8b-8192",  
    temperature=0
)

prompt = PromptTemplate(
    template='answer the following question \n {question} from the following text - \n {text}',
    input_variables=['question','text']
)

parser = StrOutputParser()

url = 'https://www.flipkart.com/apple-macbook-pro-m3-8-gb-1-tb-ssd-macos-sonoma-mtl83hn-a/p/itmcd041a34ee857?pid=COMGUTX7ENVNSZGS&lid=LSTCOMGUTX7ENVNSZGSN2DRNS&marketplace=FLIPKART&q=macbook+pro&store=6bo%2Fb5g&spotlightTagId=default_TrendingId_6bo%2Fb5g&srno=s_1_3&otracker=search&otracker1=search&fm=Search&iid=1110ba0f-f9bb-4227-8911-1372c0dd6866.COMGUTX7ENVNSZGS.SEARCH&ppt=sp&ppn=sp&ssid=m6jtgvnuv40000001754140278137&qH=9cf36a97583bd48f'


loader = WebBaseLoader(url)
docs = loader.load()

chain = prompt | model | parser

result = chain.invoke({'question':'what is the name of the seller','text':docs[0].page_content})

print(result)   