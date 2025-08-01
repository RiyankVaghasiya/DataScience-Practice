from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field
from typing import Literal
from langchain.schema.runnable import RunnableBranch, RunnableLambda


load_dotenv()

model = ChatGroq(
    model="llama3-8b-8192"
)

parser = StrOutputParser()

class Feedback(BaseModel):

    sentiment:Literal['positive','negative'] = Field(description="give the sentiment of the feedback")


parser2 = PydanticOutputParser(pydantic_object=Feedback)

prompt1 = PromptTemplate(
    template='Classify the sentiment of the following feedback text into postive or negative \n {feedback} \n {format_instruction}',
    input_variables=['feedback'],
    partial_variables={'format_instruction':parser2.get_format_instructions()}
)

classifier_chain = prompt1 | model | parser2

prompt2 = PromptTemplate(
    template='Write an appropriate response to this positive feedback \n {feedback}',
    input_variables=['feedback']
)

prompt3 = PromptTemplate(
    template='Write an appropriate response to this negative feedback \n {feedback}',
    input_variables=['feedback']
)

branch_chain = RunnableBranch(
    (lambda x:x['sentiment'] =='positive', prompt2 | model | parser),
    (lambda x:x['sentiment'] == 'negative', prompt3 | model | parser),
    RunnableLambda(lambda x: "could not find sentiment")
)

# chain = classifier_chain | branch_chain
chain = classifier_chain | RunnableLambda(lambda x: {'sentiment': x.sentiment, 'feedback': x}) | branch_chain

result = chain.invoke({'feedback': 'This is a terrible phone'})

print(result)   