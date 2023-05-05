from argparse import ArgumentParser
import openai
import os
from hashlib import sha256
import logging
import numpy as np
from typing import List
from tqdm import tqdm
from joblib import Parallel, delayed


def do_hash(text: str) -> str:
    return sha256(text.encode('utf-8')).hexdigest()


class EmbExtractor:
    def __init__(self, model='text-embedding-ada-002', storage='./emb_store'):
        openai.api_key = os.getenv("OPENAI_API_KEY")
        self._storage = storage
        self.model = model

    def _check_cache(self, text: str) -> bool:
        a_hash = do_hash(text)
        the_dir = os.path.join(self._storage, a_hash[:2], a_hash[2:4])
        return os.path.exists(os.path.join(the_dir, f'{a_hash}.npy'))

    def _save_to_cache(self, text: str, data: List[float]) -> np.ndarray:
        a_hash = do_hash(text)
        if not os.path.exists(self._storage):
            os.mkdir(self._storage)
        the_dir = os.path.join(self._storage, a_hash[:2])
        if not os.path.exists(the_dir):
            os.mkdir(the_dir)
        the_dir = os.path.join(the_dir, a_hash[2:4])
        if not os.path.exists(the_dir):
            os.mkdir(the_dir)
        np_data = np.array(data)
        np.save(os.path.join(the_dir, a_hash), np_data)
        return np_data

    def _load_from_cache(self, text: str) -> np.ndarray:
        a_hash = do_hash(text)
        the_dir = os.path.join(self._storage, a_hash[:2], a_hash[2:4])
        return np.load(os.path.join(the_dir, f'{a_hash}.npy'))

    def get_emb_for_text(self, text: str) -> np.ndarray:
        if self._check_cache(text):
            return self._load_from_cache(text)
        else:
            embeddings = openai.Embedding.create(model=self.model, input=text)
            return self._save_to_cache(text, embeddings['data'][0]['embedding'])


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--model', '-m', default='text-embedding-ada-002', help='model for embeddings')
    parser.add_argument('--storage', '-s', default='./emb_store')
    parser.add_argument('--nthreads', '-j', type=int, default=25, help='parallelization')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--file', '-f', help='file to read input from')
    group.add_argument('--text', '-t', help='text as input')
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, handlers=[logging.StreamHandler()])
    logger = logging.getLogger(__name__)

    extractor = EmbExtractor(args.model, args.storage)
    if args.text is not None:
        extractor.get_emb_for_text(args.text)
    else:
        lines = {line.strip() for line in open(args.file, 'r') if len(line.strip()) > 0}
        Parallel(n_jobs=args.nthreads)(delayed(extractor.get_emb_for_text)(x)
                                       for x in tqdm(lines, unit='line', total=len(lines)))
