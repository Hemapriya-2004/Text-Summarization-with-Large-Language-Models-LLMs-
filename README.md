# Text-Summarization with Large Language Models(LLM)
## Aim: 
   To explore how LLMs can enhance the accuracy and efficiency of summarizing extensive text data.
## Algorithm

1.Import the necessary libraries and load a pre-trained open-source LLM (e.g., GPT, BERT) from a repository such as Hugging Face's Transformers.

2.Clean and preprocess the text data to ensure it's in a suitable format for the model (e.g., tokenizing the text, handling special characters).

3.Feed the preprocessed text into the LLM to generate summaries. This typically involves using the model’s summarization pipeline or relevant functions.

4.Refine and format the generated summaries as needed. This may involve detokenizing the text or adjusting the length and coherence of the summaries.

5.Assess the quality of the summaries using evaluation metrics or human feedback. Optionally, fine-tune the model on specific datasets to improve performance for your use case.

## Program
```
!pip install --upgrade --quiet  langchain-openai tiktoken chromadb langchain
!pip install langchain_community
!pip install -q torch transformers accelerate bitsandbytes transformers sentence-transformers faiss-gpu
!pip install -q langchain huggingface_hub
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

model_name = "HuggingFaceH4/zephyr-7b-beta"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
    )

model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=bnb_config)
tokenizer = AutoTokenizer.from_pretrained(model_name)

from langchain.llms import HuggingFacePipeline
from transformers import pipeline

text_generation_pipeline = pipeline(
    model=model,
    tokenizer=tokenizer,
    task="text-generation",
    temperature=0.2,
    do_sample=True,
    repetition_penalty=1.1,
    return_full_text=True,
    max_new_tokens=400,
)

llm = HuggingFacePipeline(pipeline=text_generation_pipeline)
!pip install pypdf

from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("/content/R Hemapriya resume.pdf")
docs = loader.load()

# Check if the issue is resolved. If not, you may need to investigate further
# to determine the acceptable content types for the server.
print(docs)
print(len(docs))
docs[0]
from langchain.chains.summarize import load_summarize_chain
chain = load_summarize_chain(llm, chain_type="stuff")
chain.run(docs)
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains.llm import LLMChain
from langchain.prompts import PromptTemplate

# Define prompt
prompt_template = """Write a concise summary of the following:
"{text}"
CONCISE SUMMARY:"""
prompt = PromptTemplate.from_template(prompt_template)

# Define LLM chain

llm_chain = LLMChain(llm=llm, prompt=prompt)

# Define StuffDocumentsChain
stuff_chain = StuffDocumentsChain(llm_chain=llm_chain, document_variable_name="text")

docs = loader.load()
print(stuff_chain.run(docs))
!pip install pypdf
from langchain.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("/content/fundamental rights.pdf")
docs = loader.load()
print(len(docs))
docs[0]
```
## Output:
![image](https://github.com/user-attachments/assets/0f8b415b-59e5-4b49-9864-0d27382e0b99)

![image](https://github.com/user-attachments/assets/8cd6b809-b222-466b-9603-3df68e1b8b14)

![image](https://github.com/user-attachments/assets/507e1773-168e-4cbc-9bc6-3efd25315a2e)

![image](https://github.com/user-attachments/assets/95b79dfd-c0c3-4e69-a8af-36a4f79e27b0)

## Result:





