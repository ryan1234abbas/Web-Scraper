from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatMessagePromptTemplate, ChatPromptTemplate

template = (
    "You are tasked with extracting specific information from the following text content: {dom_content}.\n"
    "Follow these instructions:\n"
    "1. Extract information that matches directly with provided description: {parse_description}\n"
    "2. Do not include additional text, comments, or explanation in your responses.\n"
    "3. If no information matches the description, return an empty string ('').\n"
    "4. Your output should only consist of the data that is explicitly asked for."
)

model = OllamaLLM(model="llama3.2:latest")

def parse_with_ollama(dom_chunks, parse_description):
    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | model

    parsed_results = []

    for i, chunk in enumerate(dom_chunks, start=1):
        response = chain.invoke(
            {"dom_content": chunk, "parse_description": parse_description}
        )
        print(f"Parsed batch: {i} of {len(dom_chunks)}")
        parsed_results.append(response)

    return "\n".join(parsed_results)
