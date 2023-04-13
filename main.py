import os
import sys
import openai
from argparse import ArgumentParser
import logging
from rich.markdown import Markdown, Console, Rule
from prompt_toolkit import prompt
import json


VERSION = '1.0.0'


def get_system_prompt():
    return [{'role': 'system',
             'content': 'You are a helpful assistant. You help editing code and '
                        'suggest improvements. Try keep answers short, but informative. '
                        'Format output so that it is displayed nicely on a terminal with 120 columns'}]


def print_text(text, is_user):
    if is_user:
        console.print(Markdown(text), style='green')
        console.print(Rule())
    else:
        console.print(Markdown(text))
        console.print(Rule())


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, handlers=[logging.StreamHandler()])
    logger = logging.getLogger(__name__)
    logger.info(f'Version {VERSION}')
    openai.api_key = os.getenv("OPENAI_API_KEY")
    models = openai.Model.list()
    available_models = sorted([m['id'] for m in models['data']])

    parser = ArgumentParser()
    parser.add_argument('--model', '-m', choices=available_models, default='gpt-3.5-turbo')
    args = parser.parse_args()

    logger.info(f'Using {args.model} model')

    current_session_tokens = 0
    history = get_system_prompt()
    console = Console(width=120)
    try:
        while True:
            message = prompt(f'[{args.model} T{current_session_tokens}]>> ')
            if message.startswith('/'):
                if message == '/exit':
                    logger.info('Done!')
                    sys.exit()
                elif message == '/reset':
                    logger.info('Clearing conversation')
                    history = get_system_prompt()
                elif message.startswith('/save '):
                    _, file_name = message.split()
                    if not os.path.exists('convos'):
                        os.mkdir('convos')
                    with open(os.path.join('convos', f'{file_name}.json'), 'w') as f:
                        json.dump(history, f)
                else:
                    commands = ['/exit', '/reset', '/save NAME']
                    print_text(f'Available commands: {", ".join(commands)}', True)
            else:
                print_text(message, True)
                history.append({'role': 'user', 'content': message})
                response = openai.ChatCompletion.create(
                    model=args.model, messages=history
                )
                response_text = response['choices'][0]['message']
                history.append(response_text)
                current_session_tokens += response['usage']['total_tokens']
                print_text(response_text['content'], False)
    except (KeyboardInterrupt, EOFError):
        logger.info('Done!')
