from langchain_experimental.llms.ollama_functions import OllamaFunctions
from langchain.chains import create_extraction_chain
import json
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from tqdm import tqdm
from PIL import Image
import pytesseract

prompt =ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a top-tier algorithm for extracting information from text. "
            "Only extract information that is relevant to the provided text. "
            "If no information is relevant, use the schema and output "
            "an empty list where appropriate."
        ),
        ("user",
            "I need to extract information from "
            "the following text: ```\n{text}\n```\n",
        ),
    ]
)
# Schema
schema = {
  "type": "object",
  "title": "Identity Information Extractor",
  "$schema": "http://json-schema.org/draft-07/schema#",
  "required": [
    "recipes"
  ],
  "properties": {
    "identity": {
      "type": "array",
      "items": {
        "type": "object",
        "required": [
          "name",
          "id-num",
          "birth",
          "address",
          "sex"
        ],
        "properties": {
          "name": {
            "type": "string",
            "description": "The name of the identity card name."
          },
          "id-num": {
            "type": "string",
            "description": "The identity card number."
          },
          "birth": {
            "type": "string",
            "description": "The birth date of the identity card owner."
          },
          "address": {
            "type": "string",
            "description": "The address of the identity card owner."
          },
          "sex": {
            "type": "string",
            "description": "The sex of the identity card owner."
          }
        }
      }
    }
  },
  "description": "Schema for extracting identity information from text."
}

img = Image.open('2.png')

text_data = pytesseract.image_to_string(img, lang='chi_sim')
documents = text_data

# Run chain
llm = OllamaFunctions(model="llama3", temperature=0)
# llm = OllamaFunctions(model="mistral:7b-instruct", temperature=0)
chain = prompt | create_extraction_chain(schema, llm)

responses = []
input_data = {
        "text": documents,
        "json_schema": schema,  
        "instruction": (
            "ID information extraction: Chinese ID information exists within this text, including name, gender, date of birth, 18-digit ID number, and residential address."
        )
    }
# print(input_data)
response = chain.invoke(input_data)
responses.append(response)


for response in responses:
    result = response['text']
    print(json.dumps(result, indent=4, ensure_ascii=False))
    