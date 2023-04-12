import os
import sys
import openai
from argparse import ArgumentParser
import logging
from rich import print
from rich.markdown import Markdown
from prompt_toolkit import prompt, styles


def get_system_prompt():
    return [{'role': 'system',
             'content': 'You are a helpful assistant. You help editing code and '
                        'suggest improvements. Try keep answers short, but informative. '
                        'Format output so that it is displayed nicely on a terminal with 120 columns'}]


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, handlers=[logging.StreamHandler()])
    logger = logging.getLogger(__name__)
    openai.api_key = os.getenv("OPENAI_API_KEY")
    models = openai.Model.list()
    available_models = sorted([m['id'] for m in models['data']])

    parser = ArgumentParser()
    parser.add_argument('--model', '-m', choices=available_models, default='gpt-3.5-turbo')
    args = parser.parse_args()

    logger.info(f'Using {args.model} model')

    current_session_tokens = 0
    history = get_system_prompt()
    while True:
        message = prompt(f'[{args.model} T{current_session_tokens}]>> ')
        if message == '/exit':
            logger.info('Done!')
            sys.exit()
        elif message == '/reset':
            logger.info('Clearing conversation')
            history = get_system_prompt()
        history.append({'role': 'user', 'content': message})
        response = openai.ChatCompletion.create(
            model=args.model, messages=history
        )
        response_text = response['choices'][0]['message']
        history.append(response_text)
        current_session_tokens += response['usage']['total_tokens']
        print(Markdown(response_text['content']))
