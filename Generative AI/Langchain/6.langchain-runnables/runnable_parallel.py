from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableParallel, RunnableSequence
import os

load_dotenv()


model = ChatGroq(
    model="llama3-8b-8192",  
    temperature=0
)


prompt1 = PromptTemplate(
    template="generate a tweet on {topic}",
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template="generate a linkedin post on {topic}",
    input_variables=['topic']
)

parser = StrOutputParser() 

parallel_chain = RunnableParallel({
    'tweet': RunnableSequence(prompt1,model,parser),
    'linkedin': RunnableSequence(prompt2,model,parser)  
})

result = parallel_chain.invoke({'topic':'AI'})
print(result['linkedin'])