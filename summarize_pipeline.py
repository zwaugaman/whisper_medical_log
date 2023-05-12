import os
from dotenv import load_dotenv

load_dotenv()
#os.environ["LANGCHAIN_HANDLER"] = "langchain"
#os.environ["LANGCHAIN_SESSION"] = "Session 4"

from langchain import OpenAI, PromptTemplate, LLMChain
from langchain.llms import OpenAIChat
from langchain.text_splitter import TokenTextSplitter
from langchain.prompts import PromptTemplate
from langchain.chains.summarize import load_summarize_chain, _load_map_reduce_chain
from custom_chains import LoadingCallable_Map
from langchain.chains.api import tmdb_docs
from langchain.chains.api.prompt import API_RESPONSE_PROMPT
from langchain.chains import APIChain, TransformChain, SimpleSequentialChain
import tiktoken
import time
import concurrent.futures
from langchain.callbacks import get_openai_callback
from langchain import HuggingFaceHub

# Added ChatOpenAI LLM for Chat-GPT-Turbo Models. Chat-GPT Turbo uses a different model structure than the other GPT-3 models.
from langchain.chat_models import ChatOpenAI

# Added LangChain ChatOpenAI prompt libraries
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)

def open_file(filepath):
        with open(filepath, 'r', encoding='utf-8') as infile:
            return infile.read()

# create a GPT-3 encoder instance
enc = tiktoken.get_encoding("gpt2")

def _tiktoken_encoder(text: str) -> int:
    return len(enc.encode(text))

token_budget = 4000
chunk_overlap = 120
token_sizing_input = 1675
token_sizing_output = 650
model_options = ["text-curie-001","text-davinci-003","gpt-3.5-turbo"] 

configs_refine = {
    "text-curie-001": {
        "model_name": "text-curie-001",
        "token_budget": 2049,
        "chunk_overlap": 15,
        "buffer": 5,
        "token_sizing_input": 725,
        "token_sizing_output": 550,
        "time_per_run": 1.6,
        "cost_per_1k": 0.002,
    },
    "text-davinci-003": {
        "model_name": "text-davinci-003",
        "token_budget": 4000,
        "buffer": 5,
        "chunk_overlap": 140,
        "token_sizing_input": 1500,
        "token_sizing_output": 1000,
        "time_per_run": 9,
        "cost_per_1k": 0.02,
    },
    "gpt-3.5-turbo": {
        "model_name": "gpt-3.5-turbo",
        "token_budget": 4096,
        "buffer": 5,
        "chunk_overlap": 140,
        "token_sizing_input": 1500,
        "token_sizing_output": 1000,
        "time_per_run": 15,
        "cost_per_1k": 0.002,
    }
}

# Reduced the token budget for the gpt-3.5 chain.
configs_map_reduce = {
    "text-curie-001": {
        "model_name": "text-curie-001",
        "token_budget": 2049,
        "chunk_overlap": 0,
        "buffer": 2,
        "token_sizing_input": 1850,
        "token_sizing_output": 100,
        "time_per_run": 1.6,
        "cost_per_1k": 0.002,
        'combine_sizing_input': 1450,
        'combine_sizing_output': 550,
        'temperature': 0.2,
    },
    "text-davinci-003": {
        "model_name": "text-davinci-003",
        "token_budget": 4000,
        "buffer": 2,
        "chunk_overlap": 5,
        "token_sizing_input": 3500,
        "token_sizing_output": 300,
        "time_per_run": 9,
        "cost_per_1k": 0.02,
        'combine_sizing_input': 3000,
        'combine_sizing_output': 1000,
        'temperature': 0.2,
    },
    "gpt-3.5-turbo": {
        "model_name": "gpt-3.5-turbo",
        "token_budget": 4096,
        "buffer": 100,
        "chunk_overlap": 5,
        "token_sizing_input": 3000,
        "token_sizing_output": 330,
        "time_per_run": 15,
        "cost_per_1k": 0.002,
        'combine_sizing_input': 2800,
        'combine_sizing_output': 1000,
        'temperature': 0.2,
    },
    "camel-5b-hf": {
        "model_name": "camel-5b-hf",
        "token_budget": 4096,
        "buffer": 100,
        "chunk_overlap": 5,
        "token_sizing_input": 3000,
        "token_sizing_output": 330,
        "time_per_run": 15,
        "cost_per_1k": 0.002,
        'combine_sizing_input': 2800,
        'combine_sizing_output': 1000,
        'temperature': 0.2,
    }
}


def question_prompt(prompt_template = open_file('prompts/refine/prompt_template.txt')):
    question_prompt = PromptTemplate(
        template=prompt_template, 
        input_variables=["text"]
    )

    return(question_prompt)

def extract_prompt(prompt_template = open_file('prompts/extract/extract_prompt.txt')):
    extract_prompt = PromptTemplate(
        template=prompt_template, 
        input_variables=["field", "text"]
    )

    return(extract_prompt)

def refine_prompt(refine_template = open_file('prompts/refine/refine_template.txt')):
    refine_prompt = PromptTemplate(
        template=refine_template,
        input_variables=["existing_answer", "text"],
    )

    return(refine_prompt)

def map_prompt(prompt_template = open_file('prompts/map_reduce/map_prompt.txt')):
    map_prompt = PromptTemplate(
        template=prompt_template, 
        input_variables=["text"]
    )

    return(map_prompt)

def combine_prompt(prompt_template = open_file('prompts/map_reduce/combine_prompt.txt')):
    combine_prompt = PromptTemplate(
        template=prompt_template, 
        input_variables=["text"]
    )

    return(combine_prompt)

#Changed gpt-3.5-turbo LLM to ChatOpenAI
def summarize_script(script, completions_model, question=question_prompt(), refine=refine_prompt(), run_limit=3):
    if completions_model == "gpt-3.5-turbo":
        llm = ChatOpenAI(
            temperature=configs_map_reduce[completions_model]["temperature"], 
            model_name=configs_map_reduce[completions_model]["model_name"], 
            openai_api_key=os.environ['OPENAI_API_KEY'],
            max_tokens=configs_map_reduce[completions_model]["token_sizing_output"],
            request_timeout=2000,

        )
    else:
        llm = ChatOpenAI(
            temperature=configs_map_reduce[completions_model]["temperature"], 
            model_name=configs_map_reduce[completions_model]["model_name"], 
            openai_api_key=os.environ['OPENAI_API_KEY'],
            max_tokens=configs_map_reduce[completions_model]["token_sizing_output"],
        )
        
    text_splitter = TokenTextSplitter(\
        chunk_size=configs_refine[completions_model]["token_sizing_input"], 
        chunk_overlap=configs_refine[completions_model]["chunk_overlap"], 
    )

    # Start timer
    start_time = time.time()

    # Code to be executed
    texts = text_splitter.split_text(script)

    from langchain.docstore.document import Document

    print(len(texts))

    docs = [Document(page_content=t) for t in texts[:min(run_limit,len(texts))]]

    # Set Prompt for GPT-3.5-turbo
    if completions_model == "gpt-3.5-turbo":
        system_message_prompt = system_message_prompt = SystemMessagePromptTemplate.from_template("")
        human_message_prompt = HumanMessagePromptTemplate(prompt=question)
        question = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])
    else:
        question = question

    # Set Prompt for GPT-3.5-turbo
    if completions_model == "gpt-3.5-turbo":
        system_message_prompt = system_message_prompt = SystemMessagePromptTemplate.from_template("")
        human_message_prompt = HumanMessagePromptTemplate(prompt=refine)
        refine = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])
    else:
        refine = refine

    chain = load_summarize_chain(
        llm, 
        chain_type="refine",
        return_intermediate_steps=True, 
        question_prompt=question, 
        refine_prompt=refine,
    )
    
    output = chain({"input_documents": docs}, )
    
    # Stop timer
    end_time = time.time()

    # Calculate and print elapsed time
    elapsed_time = end_time - start_time
    print("Elapsed time: ", elapsed_time, " seconds")

    return(output)

def summarize_script_map_reduce(script, completions_model, max_tokens= None, chunk_size= None, chunk_overlap= None, OpenAI_api_key=os.environ.get("OPENAI_API_KEY"), map= map_prompt(), combine= combine_prompt(), run_limit= 3):
    # If the user does not provide a max_tokens value, use the default from the config
    if max_tokens is None:
        max_tokens = configs_map_reduce[completions_model]["token_sizing_output"]

    # If the user does not provide a chunk_size value, use the default from the config
    if chunk_size is None:
        chunk_size = configs_map_reduce[completions_model]["token_sizing_input"]

    # If the user does not provide a chunk_overlap value, use the default from the config
    if chunk_overlap is None:
        chunk_overlap = configs_map_reduce[completions_model]["chunk_overlap"]
    
    #Changed gpt-3.5-turbo LLM to ChatOpenAI
    if completions_model == "gpt-3.5-turbo":
        llm = ChatOpenAI(
            temperature=configs_map_reduce[completions_model]["temperature"], 
            model_name=configs_map_reduce[completions_model]["model_name"], 
            openai_api_key=OpenAI_api_key,
            max_tokens=max_tokens,
            request_timeout=2000,
        )

        llm_combine = ChatOpenAI(
            temperature=configs_map_reduce[completions_model]["temperature"], 
            model_name=configs_map_reduce[completions_model]["model_name"], 
            openai_api_key=OpenAI_api_key,
            max_tokens=configs_map_reduce[completions_model]["combine_sizing_output"],
            request_timeout=2000,
        )
    
    elif completions_model == "camel-5b-hf":
        repo_id = "Writer/camel-5b-hf" 

        llm = HuggingFaceHub(
            repo_id=repo_id, 
            model_kwargs={"temperature":0, "max_length":330},
            verbose=True,
        )

        llm_combine = HuggingFaceHub(
            repo_id=repo_id, 
            model_kwargs={"temperature":0, "max_length":330},
            verbose=True,
        )

    else:
        llm = OpenAI(
            temperature=configs_map_reduce[completions_model]["temperature"], 
            model_name=configs_map_reduce[completions_model]["model_name"], 
            openai_api_key=OpenAI_api_key,
            max_tokens=configs_map_reduce[completions_model]["token_sizing_output"],
        )

        llm_combine = OpenAI(
            temperature=configs_map_reduce[completions_model]["temperature"], 
            model_name=configs_map_reduce[completions_model]["model_name"], 
            openai_api_key=OpenAI_api_key,
            max_tokens=configs_map_reduce[completions_model]["combine_sizing_output"],
        )

    text_splitter = TokenTextSplitter(\
        chunk_size=chunk_size, 
        chunk_overlap=chunk_overlap, 
    )

    # Start timer
    start_time = time.time()

    # Code to be executed
    texts = text_splitter.split_text(script)

    from langchain.docstore.document import Document

    print(len(texts))

    # Set Prompt for GPT-3.5-turbo
    if completions_model == "gpt-3.5-turbo":
        system_message_prompt = system_message_prompt = SystemMessagePromptTemplate.from_template("")
        human_message_prompt = HumanMessagePromptTemplate(prompt=map)
        map = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])
    else:
        map = map

    # Set Prompt for GPT-3.5-turbo
    if completions_model == "gpt-3.5-turbo":
        system_message_prompt = system_message_prompt = SystemMessagePromptTemplate.from_template("")
        human_message_prompt = HumanMessagePromptTemplate(prompt=combine)
        combine = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])
    else:
        combine = combine

    docs = [Document(page_content=t) for t in texts[:min(run_limit,len(texts))]]

    with get_openai_callback() as cb:      
        chain = load_summarize_chain(
            llm= llm,
            chain_type="map_reduce",
            reduce_llm= llm_combine,
            map_prompt= map,
            combine_prompt= combine, 
            return_intermediate_steps=True,
        )
        
        output = chain({"input_documents": docs}, )
        
        # Stop timer
        end_time = time.time()

        # Calculate and print elapsed time
        elapsed_time = end_time - start_time
        print("Map Reduce Elapsed time: ", elapsed_time, " seconds and ", cb.total_tokens, " total tokens.")

    return(output)

def summarize_script_transformation_compress(script, completions_model, OpenAI_api_key=os.environ.get("OPENAI_API_KEY"), map= map_prompt(), combine= combine_prompt(), run_limit= 3):
    
    #Changed gpt-3.5-turbo LLM to ChatOpenAI
    if completions_model == "gpt-3.5-turbo":
        llm = ChatOpenAI(
            temperature=configs_map_reduce[completions_model]["temperature"], 
            model_name=configs_map_reduce[completions_model]["model_name"], 
            openai_api_key=OpenAI_api_key,
            max_tokens=configs_map_reduce[completions_model]["token_sizing_output"],
        )

    else:
        llm = OpenAI(
            temperature=configs_map_reduce[completions_model]["temperature"], 
            model_name=configs_map_reduce[completions_model]["model_name"], 
            openai_api_key=OpenAI_api_key,
            max_tokens=configs_map_reduce[completions_model]["token_sizing_output"],
        )

    text_splitter = TokenTextSplitter(\
        chunk_size=configs_map_reduce[completions_model]["token_sizing_input"], 
        chunk_overlap=configs_map_reduce[completions_model]["chunk_overlap"], 
    )

    # Start timer
    start_time = time.time()

    # Code to be executed

    with get_openai_callback() as cb:      
        def transform_func(inputs: dict) -> dict:
            text = inputs["text"]
            shortened_text = "\n\n".join(text_splitter.split_text(text))
            return {"output_text": shortened_text}

        transform_chain = TransformChain(input_variables=["text"], output_variables=["output_text"], transform=transform_func)
        
        # Set Prompt for GPT-3.5-turbo
        if completions_model == "gpt-3.5-turbo":
            system_message_prompt = system_message_prompt = SystemMessagePromptTemplate.from_template("")
            human_message_prompt = HumanMessagePromptTemplate(prompt=map)
            map = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])
        else:
            map = map

        llm_chain = LLMChain(llm, prompt=map)

        sequential_chain = SimpleSequentialChain(chains=[transform_chain, llm_chain])

        output = sequential_chain.run(script)

        # Stop timer
        end_time = time.time()

        # Calculate and print elapsed time
        elapsed_time = end_time - start_time
        print("Elapsed time: ", elapsed_time, " seconds and ", cb.total_tokens, " total tokens.")

    return(output)


def summarize_script_map(script, completions_model, max_tokens= None, chunk_size= None, chunk_overlap= None, OpenAI_api_key=os.environ.get("OPENAI_API_KEY"), map= map_prompt(), combine= combine_prompt(), temperature=0, run_limit= 3):
    # If the user does not provide a max_tokens value, use the default from the config
    if max_tokens is None:
        max_tokens = configs_map_reduce[completions_model]["token_sizing_output"]

    # If the user does not provide a chunk_size value, use the default from the config
    if chunk_size is None:
        chunk_size = configs_map_reduce[completions_model]["token_sizing_input"]

    # If the user does not provide a chunk_overlap value, use the default from the config
    if chunk_overlap is None:
        chunk_overlap = configs_map_reduce[completions_model]["chunk_overlap"]


    #Changed gpt-3.5-turbo LLM to ChatOpenAI
    if completions_model == "gpt-3.5-turbo":
        llm = ChatOpenAI(
            temperature=temperature, 
            model_name=configs_map_reduce[completions_model]["model_name"], 
            openai_api_key=OpenAI_api_key,
            max_tokens=configs_map_reduce[completions_model]["token_sizing_output"],
        )
    else:
        llm = OpenAI(
            temperature=temperature, 
            model_name=configs_map_reduce[completions_model]["model_name"], 
            openai_api_key=OpenAI_api_key,
            max_tokens=configs_map_reduce[completions_model]["token_sizing_output"],
        )

    text_splitter = TokenTextSplitter(\
        chunk_size=configs_map_reduce[completions_model]["token_sizing_input"], 
        chunk_overlap=configs_map_reduce[completions_model]["chunk_overlap"], 
    )


    # Start timer
    start_time = time.time()

    # Code to be executed
    texts = text_splitter.split_text(script)

    from langchain.docstore.document import Document

    docs = [Document(page_content=t) for t in texts[:min(run_limit,len(texts))]]

    print(len(docs))

    # Set Prompt for GPT-3.5-turbo
    if completions_model == "gpt-3.5-turbo":
        system_message_prompt = system_message_prompt = SystemMessagePromptTemplate.from_template("")
        human_message_prompt = HumanMessagePromptTemplate(prompt=map)
        map = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])
    else:
        map = map

    with get_openai_callback() as cb:       
        chain = LLMChain(
            llm= llm,
            prompt= map
        )

        def process_document(doc):
            return chain.run(doc)

        final_results = [None] * len(docs)

        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [executor.submit(process_document, doc) for doc in docs]
            results = [None] * len(docs)
            for future in concurrent.futures.as_completed(futures):
                i = futures.index(future)
                results[i] = future.result()

        final_results = "\n\n".join(results)

        # Stop timer
        end_time = time.time()

        # Calculate and print elapsed time
        elapsed_time = end_time - start_time
        print("Map Elapsed time: ", elapsed_time, " seconds and ", cb.total_tokens, " total tokens.")

    return(final_results)
    
def summarize_script_completions(script, completions_model, OpenAI_api_key=os.environ.get("OPENAI_API_KEY"), prompt= map_prompt(), run_limit = 3, max_tokens=100, temperature=0):
    #Changed gpt-3.5-turbo LLM to ChatOpenAI
    if completions_model == "gpt-3.5-turbo":
        llm = ChatOpenAI(
            temperature=temperature, 
            model_name=configs_map_reduce[completions_model]["model_name"], 
            openai_api_key=OpenAI_api_key,
            max_tokens=configs_map_reduce[completions_model]["token_sizing_output"],
        )
    else:
        llm = OpenAI(
            temperature=temperature, 
            model_name=configs_map_reduce[completions_model]["model_name"], 
            openai_api_key=OpenAI_api_key,
            max_tokens=configs_map_reduce[completions_model]["token_sizing_output"],
        )

    # Set Prompt for GPT-3.5-turbo
    if completions_model == "gpt-3.5-turbo":
        system_message_prompt = system_message_prompt = SystemMessagePromptTemplate.from_template("")
        human_message_prompt = HumanMessagePromptTemplate(prompt=prompt)
        prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])
    else:
        prompt = prompt

    # Start timer
    start_time = time.time()
    
    with get_openai_callback() as cb:       
        chain = LLMChain(
            llm= llm,
            prompt= prompt,
        )
        
        output = chain.run(script)
        
        # Stop timer
        end_time = time.time()

        # Calculate and print elapsed time
        elapsed_time = end_time - start_time
        print("Completions Elapsed time: ", elapsed_time, " seconds and ", cb.total_tokens, " total tokens.")

    return(output)

def summarize_script_extract(field, script, completions_model, prompt= map_prompt(), run_limit = 3, max_tokens=100, temperature=0):
    #Changed gpt-3.5-turbo LLM to ChatOpenAI
    if completions_model == "gpt-3.5-turbo":
        llm = ChatOpenAI(
            temperature=temperature, 
            model_name=configs_map_reduce[completions_model]["model_name"], 
            openai_api_key=os.environ['OPENAI_API_KEY'],
            max_tokens=configs_map_reduce[completions_model]["token_sizing_output"],
        )
    else:
        llm = OpenAI(
            temperature=temperature, 
            model_name=configs_map_reduce[completions_model]["model_name"], 
            openai_api_key=os.environ['OPENAI_API_KEY'],
            max_tokens=configs_map_reduce[completions_model]["token_sizing_output"],
        )

    # Set Prompt for GPT-3.5-turbo
    if completions_model == "gpt-3.5-turbo":
        system_message_prompt = system_message_prompt = SystemMessagePromptTemplate.from_template("")
        human_message_prompt = HumanMessagePromptTemplate(prompt=prompt)
        prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])
    else:
        prompt = prompt

    # Start timer
    start_time = time.time()

    with get_openai_callback() as cb:       
        chain = LLMChain(
            llm= llm,
            prompt= prompt,
        )
        
        output = chain.run({'field': field, 'text': script})
        
        # Stop timer
        end_time = time.time()

        # Calculate and print elapsed time
        elapsed_time = end_time - start_time
        print("Extract Elapsed time: ", elapsed_time, " seconds and ", cb.total_tokens, " total tokens.")

    return(output)

def summarize_script_TMDB_chain(query, completions_model, prompt= question_prompt(), run_limit = 3, max_tokens=250, temperature=0):
    #Changed gpt-3.5-turbo LLM to ChatOpenAI
    if completions_model == "gpt-3.5-turbo":
        llm = ChatOpenAI(
            temperature=temperature, 
            model_name=configs_map_reduce[completions_model]["model_name"], 
            openai_api_key=os.environ['OPENAI_API_KEY'],
            max_tokens=configs_map_reduce[completions_model]["token_sizing_output"],
        )
    else:
        llm = OpenAI(
            temperature=temperature, 
            model_name=configs_map_reduce[completions_model]["model_name"], 
            openai_api_key=os.environ['OPENAI_API_KEY'],
            max_tokens=configs_map_reduce[completions_model]["token_sizing_output"],
        )

    # Start timer
    start_time = time.time()

    headers = {"Authorization": f"Bearer {os.environ['TMDB_BEARER_TOKEN']}"}

    chain = APIChain.from_llm_and_api_docs(
        llm, 
        tmdb_docs.TMDB_DOCS, 
        headers=headers, 
        verbose=True
    )

    output = chain.run(query) 
    
    # Stop timer
    end_time = time.time()

    # Calculate and print elapsed time
    elapsed_time = end_time - start_time
    print("Elapsed time: ", elapsed_time, " seconds")

    return(output)

if __name__ == '__main__':
    summarize_script(open_file('input.txt'), 'text-davinci-003')