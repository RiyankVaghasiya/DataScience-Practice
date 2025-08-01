from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableParallel, RunnableSequence, RunnablePassthrough, RunnableLambda

load_dotenv()

def word_count(text):
    return len(text.split())

model = ChatGroq(
    model="llama3-8b-8192",  
    temperature=0
)

prompt = PromptTemplate(
    template="write a joke about the {topic}",
    input_variables=['topic']
)

parser = StrOutputParser() 

joke_gen_chain = RunnableSequence(prompt,model,parser)

parallel_chain = RunnableParallel({
   'joke': RunnablePassthrough(),
   'word_count': RunnableLambda(word_count)
})

final_chain = RunnableSequence(joke_gen_chain,parallel_chain)

result = final_chain.invoke({'topic':'AI'})

print(result)