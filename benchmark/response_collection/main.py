import argparse


def get_model():
    if args.model == 'mistral_7b':
        from mistral import Mistral
        model = Mistral(mname=args.model)
    elif args.model == 'chatglm3_6b':
        from chatglm3_6b import ChatGLM3
        model = ChatGLM3(mname=args.model)
    elif args.model == 'gpt4_turbo':
        from gpt4_turbo import GPT4Turbo
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
