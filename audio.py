import sounddevice as sd
import soundfile as sf
import queue
import logging
import numpy as np
from uuid import uuid4
import os


class AudioRecorder:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.sample_rate = 16000
        self.channels = 1

    def _record_until_key(self):
        # Create a thread-safe queue to hold the recordings
        recording_queue = queue.Queue()
        self.logger.info("Recording...")

        def callback(indata, frames, time, status):
            # Add recorded audio to the queue
            recording_queue.put(indata.copy())

        with sd.InputStream(samplerate=self.sample_rate, channels=self.channels, callback=callback):
            # Wait for key to be pressed, keyboard.wait doesn't work on Mac
            input('Press enter to stop recording...')

        self.logger.info("Done recording")

        # Combine all recorded audio into one array
        recording = []
        while not recording_queue.empty():
            recording.append(recording_queue.get())
        recording = np.concatenate(recording)
        self.logger.info(f'Recorded {recording.size / self.sample_rate:2.1f} seconds')
        return recording

    def save(self, recording, file_name=None):
        if file_name is None:
            file_name = str(uuid4()) + '.wav'
        if not os.path.exists('audios'):
            os.mkdir('audios')
        file_path = os.path.join('audios', file_name)
        sf.write(file_path, recording, self.sample_rate)
        self.logger.info(f'Audio stored to {file_path}')
        return file_path

    def record(self):
        return self._record_until_key()
