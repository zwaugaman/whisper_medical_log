import openai
import summarize_pipeline
import os
from io import BytesIO
from werkzeug.datastructures import FileStorage


def audio_log(audio_data):
    file_storage = BytesIO(audio_data)
    file_storage.name = "audio.webm" # fake file name

    print(file_storage)

    transcript = openai.Audio.transcribe("whisper-1", file_storage)

    transcript = transcript["text"]
    
    print(transcript)

    # Summarize the transcript as Story
    memories_map_template = """Instructions: Combine the details below. Each unique detail must be on a separate line. The combined output must be less than 4,000 characters long. Keep the content and context preserved.

                            Input: {text} 

                            Output:"""
    
    memories_combine_template = """Instructions: Combine the memories below. Each unique memory must be on a separate line. The combined output must be less than 4,000 characters long. Keep the content and context preserved.

                            Input: {text} 

                            Output:"""
    
    transcript_summary = summarize_pipeline.summarize_script_map_reduce( \
        transcript, \
        "gpt-3.5-turbo", \
        os.environ.get("OPENAI_API_KEY"), \
        summarize_pipeline.map_prompt(memories_map_template), \
        summarize_pipeline.combine_prompt(memories_combine_template),
        500 \
    ) 

    # Create a Title
    title_map_template = """Instructions: You have been given a section of a transcript of an oral history. I want you to act as a title generator for written pieces. Please keep the title concise and under 20 words, and ensure that the meaning is maintained. Replies will utilize the language type of the topic. Extract the date as well. Do not write any explanations or other words, just reply with the title and date. 

                        Input: {text}

                        Output:"""
    
    title_combine_template = """Instructions: You have been given a list of titles. I want you to act as a title generator for written pieces. Combine these titles into one. Please keep the title concise and under 20 words, and ensure that the meaning is maintained. Replies will utilize the language type of the topic. Do not write any explanations or other words, just reply with the title. 

                        Input: {text}

                        Output:"""

    title = summarize_pipeline.summarize_script_map_reduce( \
        transcript_summary['output_text'], \
        "gpt-3.5-turbo", \
        os.environ.get("OPENAI_API_KEY"), \
        summarize_pipeline.map_prompt(title_map_template), \
        summarize_pipeline.combine_prompt(title_combine_template),
        500 \
    ) 

    # Output String Format
    title = title['output_text']
    memories = transcript_summary['output_text']

    return f"Title: {title}" + f"\nMemories:\n{memories}"

