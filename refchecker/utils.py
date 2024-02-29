import time
import json
import os

import spacy
import boto3
import openai
from openai import RateLimitError as OpenAIRateLimitError
from openai import APIError as OpenAIAPIError
from openai import Timeout as OpenAITimeout
import anthropic
from anthropic import HUMAN_PROMPT, AI_PROMPT
from litellm import batch_completion


# Setup spaCy NLP
nlp = None

# Setup OpenAI API
openai_client = None

# Setup Claude 2 API
bedrock = None
anthropic_client = None


def sentencize(text):
    """Split text into sentences"""
    global nlp
    if not nlp:
        nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)
    return [sent for sent in doc.sents]


def split_text(text, segment_len=200):
    """Split text into segments according to sentence boundaries."""
    segments, seg = [], []
    sents = [[token.text for token in sent] for sent in sentencize(text)]
    for sent in sents:
        if len(seg) + len(sent) > segment_len:
            segments.append(" ".join(seg))
            seg = sent
            # single sentence longer than segment_len
            if len(seg) > segment_len:
                # split into chunks of segment_len
                seg = [
                    " ".join(seg[i:i+segment_len])
                    for i in range(0, len(seg), segment_len)
                ]
                segments.extend(seg)
                seg = []
        else:
            seg.extend(sent)
    if seg:
        segments.append(" ".join(seg))
    return segments


def get_model_batch_response(
        prompts,
        temperature=0,
        model='gpt-3.5-turbo',
        n_choices=1,
        max_new_tokens=500
):
    """
    Get batch generation results with given prompts.

    Parameters
    ----------
    prompts : List[str]
        List of prompts for generation.
    temperature : float, optional
        The generation temperature, use greedy decoding when setting
        temperature=0, defaults to 0.
    model : str, optional
        The model for generation, defaults to 'gpt-4'.
    n_choices : int, optional
        How many samples to return for each prompt input, defaults to 1.
    max_new_tokens : int, optional
        Maximum number of newly generated tokens, defaults to 500.

    Returns
    -------
    response_list : List[str]
        List of generated text.
    """

    if not prompts or len(prompts) == 0:
        raise ValueError("Invalid input.")

    message_list = []
    for prompt in prompts:
        if len(prompt) == 0:
            raise ValueError("Invalid prompt.")
        if isinstance(prompt, str):
            messages = [{
                'role': 'user',
                'content': prompt
            }]
        elif isinstance(prompt, list):
            messages = prompt
        else:
            raise ValueError("Invalid prompt type.")
        message_list.append(messages)
    import litellm
    litellm.suppress_debug_info = True
    while True:
        try:
            responses = batch_completion(
                model=model,
                messages=message_list,
                temperature=temperature,
                n=n_choices,
                max_tokens=max_new_tokens
            )
            if n_choices == 1:
                response_list = [r.choices[0].message.content for r in responses]
            else:
                response_list = [[res.message.content for res in r.choices] for r in responses]
            for r in response_list:
                if not r or len(r) == 0:
                    raise ValueError(f'{model} API returns None or empty string')
            return response_list
        except Exception as e:
            if isinstance(e, OpenAIRateLimitError) or isinstance(e, OpenAIAPIError) or isinstance(e, OpenAITimeout):
                print(f"{e} [sleep 10 seconds]")
                time.sleep(10)
                continue
            print(e)
            return None

def get_openai_model_response(
    prompt, 
    temperature=0, 
    model='gpt-3.5-turbo', 
    n_choices=1,
    max_new_tokens=500
):
    global openai_client
    if not openai_client:
        openai_client = openai.OpenAI()
    
    if not prompt or len(prompt) == 0:
        return None

    while True:
        try:
            if isinstance(prompt, str):
                messages = [{
                    'role': 'user',
                    'content': prompt
                }]
            elif isinstance(prompt, list):
                messages = prompt
            else:
                return None

            res_choices = openai_client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    n=n_choices,
                    max_tokens=max_new_tokens
                ).choices
            if n_choices == 1:
                response = res_choices[0].message.content
            else:
                response = [res.message.content for res in res_choices]

            if response and len(response) > 0:
                return response
        except Exception as e:
            if isinstance(e, OpenAIRateLimitError) or isinstance(e, OpenAIAPIError) or isinstance(e, OpenAITimeout):
                time.sleep(10)
                continue
            print(type(e), e)
            return None
        return None


def get_claude2_response(prompt, temperature=0, max_new_tokens=500):
    if os.environ.get('aws_bedrock_region'):
        global bedrock
        if not bedrock:
            bedrock = boto3.client(
                service_name='bedrock-runtime',
                region_name=os.environ.get('aws_bedrock_region')
            )
        return _get_bedrock_claude_completion(
            prompt=prompt,
            temperature=temperature,
            max_new_tokens=max_new_tokens
        )
    else:
        global anthropic_client
        if not anthropic_client:
            anthropic_client = anthropic.Anthropic()
        return _get_anthropic_claude_completion(
            prompt=prompt,
            temperature=temperature,
            max_new_tokens=max_new_tokens
        )


def _get_bedrock_claude_completion(prompt, temperature=0, max_new_tokens=300):
    if not prompt or len(prompt) == 0:
        return None
    while True:
        try:
            body = json.dumps({
                "prompt": f"\n\nHuman: {prompt} \n\nAssistant:",
                "max_tokens_to_sample": max_new_tokens,
                "temperature": temperature,
                "top_p": 0.9,
            })
            modelId = 'anthropic.claude-v2'
            accept = 'application/json'
            contentType = 'application/json'

            response = bedrock.invoke_model(body=body, modelId=modelId, accept=accept, contentType=contentType)

            response_body = json.loads(response.get('body').read())
            # text
            return response_body.get('completion')
        except Exception as e:
            if e.response['Error']['Code'] == 'ThrottlingException':
                time.sleep(10)
                continue
            print(type(e), e)
            return None
        return None


def _get_anthropic_claude_completion(prompt, temperature=0, max_new_tokens=300):
    if not prompt or len(prompt) == 0:
        return None
    
    completion = anthropic_client.completions.create(
        model="claude-2",
        max_tokens_to_sample=max_new_tokens,
        temperature=temperature,
        prompt=f"{HUMAN_PROMPT} {prompt}{AI_PROMPT}",
    )
    return completion.completion
