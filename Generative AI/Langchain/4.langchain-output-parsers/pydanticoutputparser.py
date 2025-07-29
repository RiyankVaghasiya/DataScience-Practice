from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
import os


hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")

llm = HuggingFaceEndpoint(
    repo_id="google/gemma-2-2b-it",
    task="text-generation",
    huggingfacehub_api_token=hf_token
)

model = ChatHuggingFace(llm=llm)

class Person(BaseModel):
    
    name:str = Field(description="Name of the person")
    age:int = Field(gt=18, description="Age of the person")
    city:str = Field(description="City name of the person")

parser = PydanticOutputParser(pydantic_object=Person)

template = PromptTemplate(
    template="Generate the name, age and city of a frictional {place} person \n {format_instruction}",
    input_variables=['place'],
    partial_variables={"format_instruction": parser.get_format_instructions()}
)

# prompt = template.format(place = "indian")

# result = model.invoke(prompt)

# final_result = parser.parse(result.content)

chain = template | model | parser

final_result = chain.invoke({'place':"indian"})


print(final_result)
