import os
import sys
import openai
from argparse import ArgumentParser
import logging
from rich.markdown import Markdown, Console, Rule
from prompt_toolkit import prompt
import json

from audio import AudioRecorder


VERSION = '1.0.1'


def get_system_prompt():
    return [{'role': 'system',
             'content': 'You are a helpful assistant. You help editing code and '
                        'suggest improvements. Try keep answers short, but informative.'}]


def get_asr_prompt():
    return [
        {'role': 'system', 'content': 'Next user message is a result of running automatic speech recognition, '
                                      'so it may contain mistakes.'}
    ]


def process_user_message(message: str, history: list) -> int:
    print_text(message, True)
    history.append({'role': 'user', 'content': message})
    response = openai.ChatCompletion.create(
        model=args.model, messages=history
    )
    response_text = response['choices'][0]['message']
    history.append(response_text)
    print_text(response_text['content'], False)
    return response['usage']['total_tokens']


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
                elif message.startswith('/load '):
                    _, file_name = message.split()
                    file_path = os.path.join('convos', f'{file_name}.json')
                    if not os.path.exists(file_path):
                        logger.warning('Conversation not found')
                    else:
                        with open(os.path.join('convos', f'{file_name}.json'), 'r') as f:
                            history = json.load(f)
                        for el in history:
                            if el['role'] == 'user':
                                print_text(el['content'], True)
                            elif el['role'] == 'assistant':
                                print_text(el['content'], False)
                elif message == '/asr':
                    recorder = AudioRecorder()
                    audio_data = recorder.record()
                    file_name = recorder.save(audio_data)
                    with open(file_name, 'rb') as f:
                        transcript = openai.Audio.transcribe("whisper-1", f)
                    transcript_file = file_name.replace('wav', 'json')
                    with open(transcript_file, 'w') as f:
                        json.dump(transcript, f)
                    message = transcript['text']
                    history.extend(get_asr_prompt())
                    used_tokens = process_user_message(message, history)
                    current_session_tokens += used_tokens
                else:
                    commands = ['/exit', '/reset', '/save NAME', '/load NAME', '/asr']
                    print_text(f'Available commands: {", ".join(commands)}', True)
            else:
                used_tokens = process_user_message(message, history)
                current_session_tokens += used_tokens
    except (KeyboardInterrupt, EOFError):
        logger.info('Done!')
