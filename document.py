import PyPDF2
import logging
from tqdm import tqdm
from typing import List
import openai
import os
import requests


DEFAULT_CONTEXT_LENGTH = 1800
logger = logging.getLogger(__name__)
openai.api_key = os.getenv("OPENAI_API_KEY")


class EmbDocument:
    def __init__(self, text: str, max_context: int = DEFAULT_CONTEXT_LENGTH, model: str = 'gpt-3.5-turbo'):
        self._text = text.split()
        self._max_context = max_context
        self._model = model
        self._summary = None
        self.used_tokens = 0

    def _prepare_chunk(self, text: List[str]):
        messages = [{'role': 'system',
                     'content': 'You are analyzing scientific papers and providing summaries of them. '
                                'Document is split in parts that are sent to you one by one.'}]
        if self._summary is not None:
            messages.append({'role': 'user',
                             'content': f'This is your summary of document so far:\n{self._summary}'})
            messages.append({'role': 'user',
                             'content': 'Provide summary of the whole document, keep it withing 1000 words. '
                                        'Do not summarize each part individually, just update you current summary '
                                        'based on information in new part of the document.'})
        messages.append({'role': 'user', 'content': f'New part of document:\n{" ".join(text)}'})
        response = openai.ChatCompletion.create(
            model=self._model, messages=messages
        )
        response_text = response['choices'][0]['message']
        self._summary = response_text['content']
        self.used_tokens += response['usage']['total_tokens']

    @property
    def summary(self):
        if self._summary is None:
            handle = tqdm(range(0, len(self._text), self._max_context), total=len(self._text), desc='Summarizing')
            for i in handle:
                chunk = self._text[i:i + self._max_context]
                self._prepare_chunk(chunk)
                handle.update(len(chunk))
        return self._summary

    @staticmethod
    def from_pdf_file(pdf_file_path: str, max_context: int = DEFAULT_CONTEXT_LENGTH, model: str = 'gpt-3.5-turbo'):
        with open(pdf_file_path, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            logger.info(f'Loaded a pdf with {len(reader.pages)} pages')
            text = ''
            for page in reader.pages:
                text += page.extract_text()
            logger.info(f'Document has {len(text.split())} words')
            return EmbDocument(text, max_context, model)

    @staticmethod
    def from_pdf_url(url: str, max_context: int = DEFAULT_CONTEXT_LENGTH, model: str = 'gpt-3.5-turbo'):
        file_path = os.path.join('docs', url)
        if not os.path.exists(file_path):
            response = requests.get(url)
            if not os.path.exists(os.path.dirname(file_path)):
                os.makedirs(os.path.dirname(file_path), exist_ok=True)
            with open(file_path, 'wb') as f:
                f.write(response.content)
        return EmbDocument.from_pdf_file(file_path, max_context, model)
