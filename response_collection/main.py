import argparse

from mistral import Mistral
from chatglm3_6b import ChatGLM3
from gpt4_turbo import GPT4Turbo



def get_model():
    if args.model == 'mistral_7b':
        model = Mistral(mname=args.model)
    elif args.model == 'chatglm3_6b':
        model = ChatGLM3(mname=args.model)
    elif args.model == 'gpt4_turbo':
        model = GPT4Turbo(mname=args.model)

    return model

def main():
    model = get_model()
    model.collect_response()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, choices=['mistral_7b', 'chatglm3_6b', 'gpt4_turbo'])
    
    args = parser.parse_args()
    
    main()
