import time
import json
import os

import spacy
from openai.types import Completion as OpenAICompletion
from openai import RateLimitError as OpenAIRateLimitError
from openai import APIError as OpenAIAPIError
from openai import Timeout as OpenAITimeout

from litellm import batch_completion
from litellm.types.utils import ModelResponse

# Setup spaCy NLP
nlp = None

# Setup OpenAI API
openai_client = None

# Setup Claude 2 API
bedrock = None
anthropic_client = None


def get_llm_full_name(llm_name):
    if llm_name == 'claude3-sonnet':
        return 'anthropic.claude-3-sonnet-20240229-v1:0' if os.environ.get('AWS_REGION_NAME') else 'claude-3-sonnet-20240229'
    elif llm_name == 'claude3-haiku':
        return 'anthropic.claude-3-haiku-20240307-v1:0' if os.environ.get('AWS_REGION_NAME') else 'claude-3-haiku-20240307'
    return llm_name


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
        model='bedrock/anthropic.claude-3-sonnet-20240229-v1:0',
        temperature=0,
        n_choices=1,
        max_new_tokens=500,
        api_base=None,
        custom_llm_api_func=None,
        **kwargs
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
        The model for generation, defaults to 'bedrock/anthropic.claude-3-sonnet-20240229-v1:0'.
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
    
    if custom_llm_api_func is not None:
        return custom_llm_api_func(prompts)
    else:
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
        # litellm.drop_params=True
        while True:
            responses = batch_completion(
                model=model,
                messages=message_list,
                temperature=temperature,
                n=n_choices,
                max_tokens=max_new_tokens,
                api_base=api_base,
                **kwargs
            )
            try:
                assert all([isinstance(r, ModelResponse) for r in responses])
                if n_choices == 1:
                    response_list = [r.choices[0].message.content for r in responses]
                else:
                    response_list = [[res.message.content for res in r.choices] for r in responses]
                
                assert all([r is not None for r in response_list])
                return response_list
            except:
                exception = None
                for e in responses:
                    if isinstance(e, ModelResponse):
                        continue
                    elif isinstance(e, OpenAIRateLimitError) or isinstance(e, OpenAIAPIError) or isinstance(e, OpenAITimeout):
                        exception = e
                        break
                    else:
                        print('Exit with the following error:')
                        print(e)
                        return None
                
                print(f"{exception} [sleep 10 seconds]")
                time.sleep(10)
                continue