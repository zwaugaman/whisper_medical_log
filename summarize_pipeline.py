from langchain import OpenAI, PromptTemplate, LLMChain
from langchain.llms import OpenAIChat
from langchain.text_splitter import TokenTextSplitter
from langchain.prompts import PromptTemplate
from langchain.chains.summarize import _load_map_reduce_chain
import time
import concurrent.futures
from langchain.callbacks import get_openai_callback

def open_file(filepath):
        with open(filepath, 'r', encoding='utf-8') as infile:
            return infile.read()

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
        "time_per_run": 9,
        "cost_per_1k": 0.002,
    }
}

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
        "buffer": 2,
        "chunk_overlap": 5,
        "token_sizing_input": 3500,
        "token_sizing_output": 300,
        "time_per_run": 9,
        "cost_per_1k": 0.002,
        'combine_sizing_input': 3000,
        'combine_sizing_output': 1000,
        'temperature': 0.2,
    }
}

def extract_prompt(prompt_template = open_file('prompts/extract/extract_prompt.txt')):
    extract_prompt = PromptTemplate(
        template=prompt_template, 
        input_variables=["field", "text"]
    )

    return(extract_prompt)

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

def summarize_script_map_reduce(script, completions_model, OpenAI_api_key, map= map_prompt(), combine= combine_prompt(), run_limit= 3):
    if completions_model == "gpt-3.5-turbo":
        llm = OpenAIChat(
            temperature=configs_map_reduce[completions_model]["temperature"], 
            model_name=configs_map_reduce[completions_model]["model_name"], 
            openai_api_key=OpenAI_api_key,
            max_tokens=configs_map_reduce[completions_model]["token_sizing_output"],
        )

        llm_combine = OpenAIChat(
            temperature=configs_map_reduce[completions_model]["temperature"], 
            model_name=configs_map_reduce[completions_model]["model_name"], 
            openai_api_key=OpenAI_api_key,
            max_tokens=configs_map_reduce[completions_model]["combine_sizing_output"],
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
        chunk_size=configs_map_reduce[completions_model]["token_sizing_input"], 
        chunk_overlap=configs_map_reduce[completions_model]["chunk_overlap"], 
    )

    # Start timer
    start_time = time.time()

    # Code to be executed
    texts = text_splitter.split_text(script)

    from langchain.docstore.document import Document

    print(len(texts))

    docs = [Document(page_content=t) for t in texts[:min(run_limit,len(texts))]]

    with get_openai_callback() as cb:      
        chain = _load_map_reduce_chain(
            llm= llm,
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
        print("Elapsed time: ", elapsed_time, " seconds and ", cb.total_tokens, " total tokens.")

    return(output)

def summarize_script_map(script, completions_model, OpenAI_api_key, map= map_prompt(), combine= combine_prompt(), temperature=0, run_limit= 3):
    if completions_model == "gpt-3.5-turbo":
        llm = OpenAIChat(
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

    print(len(texts))

    docs = [Document(page_content=t) for t in texts[:min(run_limit,len(texts))]]

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
            for i, future in enumerate(concurrent.futures.as_completed(futures)):
                final_results[i] = future.result()

        results = "".join(final_results)

        # Stop timer
        end_time = time.time()

        # Calculate and print elapsed time
        elapsed_time = end_time - start_time
        print("Elapsed time: ", elapsed_time, " seconds and ", cb.total_tokens, " total tokens.")

    return(results)
    
def summarize_script_completions(script, completions_model, OpenAI_api_key, prompt= map_prompt(), run_limit = 3, max_tokens=100, temperature=0):
    llm = OpenAI(
        temperature=temperature, 
        model_name=configs_map_reduce[completions_model]["model_name"], 
        openai_api_key=OpenAI_api_key,
        max_tokens=max_tokens,
    )

    # Start timer
    start_time = time.time()

    chain = LLMChain(
        llm= llm,
        prompt= prompt,
    )
    
    output = chain.run(script)
    
    # Stop timer
    end_time = time.time()

    # Calculate and print elapsed time
    elapsed_time = end_time - start_time
    print("Elapsed time: ", elapsed_time, " seconds")

    return(output)

def summarize_script_extract(field, script, completions_model, OpenAI_api_key, prompt= map_prompt(), run_limit = 3, max_tokens=100, temperature=0):
    llm = OpenAI(
        temperature=temperature, 
        model_name=configs_map_reduce[completions_model]["model_name"], 
        openai_api_key=OpenAI_api_key,
        max_tokens=max_tokens,
    )

    # Start timer
    start_time = time.time()

    chain = LLMChain(
        llm= llm,
        prompt= prompt,
    )
    
    output = chain.run({'field': field, 'text': script})
    
    # Stop timer
    end_time = time.time()

    # Calculate and print elapsed time
    elapsed_time = end_time - start_time
    print("Elapsed time: ", elapsed_time, " seconds")

    return(output)