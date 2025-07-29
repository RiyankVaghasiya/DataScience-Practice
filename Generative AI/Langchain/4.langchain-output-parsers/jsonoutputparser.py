from langchain_core.prompts import PromptTemplate
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
import os
from langchain_core.output_parsers import JsonOutputParser

hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")

llm = HuggingFaceEndpoint(
    repo_id="google/gemma-2-2b-it",
    task="text-generation",
    huggingfacehub_api_token=hf_token
)

model = ChatHuggingFace(llm=llm)

parser = JsonOutputParser()

template = PromptTemplate(
    template="Give me 5 fact about {topic} \n{format_instruction}",
    input_variables=['topic'],
    partial_variables={
        'format_instruction':parser.get_format_instructions()
    }
)

# prompt = template.format()

# result = model.invoke(prompt)

# final_result = parser.parse(result.content)

chain = template | model | parser
final_result = chain.invoke({'topic': 'black hole'})

print(final_result) 