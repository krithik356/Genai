import os
import time
import random
from dotenv import load_dotenv
from openai import (
    OpenAI,
    APIConnectionError,
    APITimeoutError,
    RateLimitError,
    InternalServerError,
)

load_dotenv()


PROVIDERS = [
    {
        "name": "gemini",
        "model": "gemini-2.5-flash",
        "client": OpenAI(
            api_key=os.environ["GEMINI_API_KEY"],
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
            max_retries=0,   
            timeout=30,
        ),
    },
    {
        "name": "openai",
        "model": "gpt-4o-mini",
        "client": OpenAI(
            api_key=os.environ["OPENAI_API_KEY"],
            max_retries=0,
            timeout=30,
        ),
    },
]


RETRYABLE = (RateLimitError, APITimeoutError, APIConnectionError, InternalServerError)

MAX_RETRIES = 3        
BASE_DELAY = 1.0       
MAX_DELAY = 20.0      


def call_with_retries(provider, **params):
    """Call one provider, retrying with exponential backoff + jitter."""
    for attempt in range(MAX_RETRIES + 1):
        try:
            return provider["client"].chat.completions.create(
                model=provider["model"], **params
            )
        except RETRYABLE as e:
            if attempt == MAX_RETRIES:
                raise  
            delay = min(BASE_DELAY * (2 ** attempt), MAX_DELAY)
            delay += random.uniform(0, delay * 0.25) 
            print(f"[{provider['name']}] {type(e).__name__}: retry "
                  f"{attempt + 1}/{MAX_RETRIES} in {delay:.1f}s")
            time.sleep(delay)


def chat(**params):
    """Try each provider in order; return (response, provider_name)."""
    last_error = None
    for provider in PROVIDERS:
        try:
            response = call_with_retries(provider, **params)
            return response, provider["name"]
        except Exception as e:
            print(f"[{provider['name']}] failed: {type(e).__name__}: {e}")
            last_error = e
    raise RuntimeError("All providers failed") from last_error 


response, used = chat(
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Who is Ronaldo?"},
    ],
    temperature=0.7,
    top_p=0.9,
    # max_tokens=500,
    # stop=["END", "\n\n\n"],
)
print(f"Answered by: {used}\n")
print(response.choices[0].message.content)

usage = response.usage
print("\n--- Token Usage ---")
print(f"Prompt tokens:     {usage.prompt_tokens}")
print(f"Completion tokens: {usage.completion_tokens}")
print(f"Total tokens:      {usage.total_tokens}")