from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableParallel, RunnableSequence, RunnablePassthrough
import os

load_dotenv()


model = ChatGroq(
    model="llama3-8b-8192",  
    temperature=0
)

prompt1 = PromptTemplate(
    template="write a joke about the {topic}",
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template="explain the following text - {text}",
    input_variables=['text']
)

parser = StrOutputParser() 

joke_gen_chain = RunnableSequence(prompt1,model,parser)

parallel_chain = RunnableParallel({
   'joke': RunnablePassthrough(),
   'explanation': RunnableSequence(prompt2,model,parser)
})

final_chain = RunnableSequence(joke_gen_chain,parallel_chain)

result = final_chain.invoke({'topic':'cricket'})
print(result)