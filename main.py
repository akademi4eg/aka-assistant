import os
import sys
import openai
from argparse import ArgumentParser
import logging
from rich.markdown import Markdown, Rule
from rich.console import Console
from prompt_toolkit import prompt
from prompt_toolkit.key_binding import KeyBindings
import json
from rich.segment import Segment
import base64

from audio import AudioRecorder


VERSION = '1.0.4'


kb = KeyBindings()


@kb.add('escape', 'enter')
def _(event):
    event.current_buffer.insert_text('\n')


@kb.add('enter')
def _(event):
    event.current_buffer.validate_and_handle()


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


class InlineImage:
    def __init__(self, data, lines=1, **kwargs):
        def _b64(value):
            return base64.b64encode(value).decode('ascii')

        kwargs.update(inline=1)

        if 'name' in kwargs:
            kwargs['name'] = _b64(kwargs['name'].encode('utf-8'))

        args = ';'.join(f'{k}={v}' for k, v in kwargs.items())
        move = f'\x1b[{lines-1}A' if lines > 1 else ''

        self._ctrl = f'\x1b]1337;File={args}:{_b64(data)}\a{move}'
        self._text = lines * '\n'

    def __rich_console__(self, console, options):
        yield Segment(self._ctrl, control=True)
        yield Segment(self._text)


def print_image(image_path):
    console.print(InlineImage(open(image_path, 'rb').read()))


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
    commands_list = [('/exit', 'To exit: /exit'), ('/clear', 'To clear history: /clear'),
                     ('/save', 'To save conversation: /save NAME'),
                     ('/load', 'To restore past conversation: /load NAME'), ('/asr', 'Speak instead of typing: /asr'),
                     ('/image', 'Generate an image: /image PROMPT'), ('/drop', 'Drop last message: /drop')]
    try:
        while True:
            message = prompt(f'[{args.model} T{current_session_tokens}]>> ', multiline=True, key_bindings=kb,
                             bottom_toolbar='Type text or a command: ' + ', '.join([c[0] for c in commands_list]))
            if message.startswith('/'):
                if message == '/exit':
                    logger.info('Done!')
                    sys.exit()
                elif message == '/clear':
                    logger.info('Clearing conversation')
                    history = get_system_prompt()
                elif message == '/drop':
                    logger.info('Dropping last user and assistant messages')
                    history = history[:-2]
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
                elif message.startswith('/image '):
                    parts = message.split()
                    image_prompt = ' '.join(parts[1:])
                    print_text(f'[C{len(image_prompt)}] {image_prompt}', is_user=True)
                    response = openai.Image.create(prompt=image_prompt, response_format='b64_json',
                                                   size='512x512')
                    created = response['created']
                    gen_image = base64.decodebytes(response['data'][0]['b64_json'].encode('utf-8'))
                    img_path = os.path.join('images', f'{created}.png')
                    if not os.path.exists('images'):
                        os.mkdir('images')
                    with open(img_path, 'wb') as f:
                        f.write(gen_image)
                    print_image(img_path)
                else:
                    help_str = "\n".join([c[1] for c in commands_list])
                    print_text(f'Available commands:\n{help_str}', True)
            else:
                used_tokens = process_user_message(message, history)
                current_session_tokens += used_tokens
    except (KeyboardInterrupt, EOFError):
        logger.info('Done!')
