from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate

template = (
    "You are tasked with extracting specific information from the following text content: {dom_content}.\n"
    "Instructions:\n"
    "1. Extract only what matches the description: {parse_description}\n"
    "2. Do NOT include extra commentary.\n"
    "3. If no match, return an empty string ('')."
)

model = OllamaLLM(model="llama3.2:latest")  # Make sure Ollama server is running

def parse_with_ollama(dom_chunks, parse_description):
    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | model

    results = []
    for chunk in dom_chunks:
        response = chain.invoke({
            "dom_content": chunk,
            "parse_description": parse_description
        })
        results.append(response)
    return "\n".join(results)
