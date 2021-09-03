import hashlib
import logging
import os
import pickle
import sys
from datetime import datetime, timedelta
from queue import Queue
from threading import Thread
from time import sleep, time

import librosa
import numpy as np
import requests
import sounddevice as sd
import soundfile as sf
from pushover import Client, init

from config import MODEL, SAVE_LOCATION, SR
from creds import app_token, client_token

# sd.default.device = 1
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(thread)d - %(message)s')

logger = logging.getLogger(__name__)

class PushWorker(Thread):

    def __init__(self, notif_queue):
        Thread.__init__(self)
        self.notif_queue = notif_queue

        init(app_token)
        self.members = [Client(client_token)] # Mine

        self.last_pinged = time()

    def run(self):
        logging.info('Staring Push Thread!')
        while True:
            datetimestr = self.notif_queue.get()
            if time() - self.last_pinged > 5:
                for client in self.members:
                    client.send_message(f'http://192.168.0.135:8000/assets/unlabeled_data/{datetimestr}.wav')
                try:
                    requests.get('http://192.168.0.205/bell')
                except requests.exceptions.RequestException as e:
                    logging.warning(f'Cannot connect to IP of ESP32, error: {e}')
                self.last_pinged = time()
            self.notif_queue.task_done()


class SaveWorker(Thread):

    def __init__(self, save_queue, sr, location):
        Thread.__init__(self)
        self.save_queue = save_queue
        self.location = location
        self.sr = sr

    def run(self):
        logging.info('Staring Save Thread!')
        while True:
            recording, datetimestr = self.save_queue.get()
            sf.write(os.path.join(self.location, f'{datetimestr}.wav'),
                     recording, self.sr, subtype='PCM_24')
            self.save_queue.task_done()


class RecordingWorker(Thread):

    def __init__(self, bell_queue, seconds, sr):
        Thread.__init__(self)
        self.bell_queue = bell_queue
        self.seconds = seconds
        self.sr = sr
        self.stream = sd.InputStream(channels=1, samplerate=self.sr, callback=self.audio_callback)
        self.block_queue = Queue()
        self.window_data = np.zeros((int(self.seconds * self.sr), 1))


    def audio_callback(self, indata, frames, time, status):
        """This is called (from a separate thread) for each audio block."""
        if status:
            logging.warning(status)
        # Fancy indexing with mapping creates a (necessary!) copy:
        self.block_queue.put(indata.copy())


    def run(self):
        logging.info('Staring Recording Thread!')
        prev_time = time()
        with self.stream:
            while True:
                # logging.info(f'Elapsed Time Between recordings: {time() - prev_time}')
                # Get the work from the bell_queue and expand the tuple
                # recording = sd.rec(int(self.seconds * self.sr), samplerate=self.sr, channels=1)
                while not self.block_queue.empty():
                    #logging.info(f'Block Q length: {self.block_queue.qsize()}')
                    data = self.block_queue.get()
                    shift = len(data)
                    self.window_data = np.roll(self.window_data, -shift, axis=0)
                    self.window_data[-shift:, :] = data
                    self.block_queue.task_done()
                datetimestr = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
                logging.debug(f'Putting current window in queue! {datetimestr}')
                self.bell_queue.put((self.sr, self.window_data.reshape(self.window_data.shape[0]), datetimestr))
                prev_time = time()
                sleep(self.seconds / 2)


class DetectionWorker(Thread):

    def __init__(self, bell_queue, notif_queue, save_queue):
        Thread.__init__(self)
        self.bell_queue = bell_queue
        self.notif_queue = notif_queue
        self.save_queue = save_queue
        # Load the Sklearn model.
        self.model = pickle.load(open(MODEL, 'rb'))
        self.model_hash = hashlib.md5(open(MODEL,'rb').read()).hexdigest()
        logging.info(f'Loaded model with hash: {self.model_hash}')
        self.last_updated = datetime.now()
        self.model_check_interval = 1  # minutes


    def process_recording(self, signal, sr):

        if sr == 0:
            # audio file IO problem
            return -1, -1, -1
        X = signal.T

        mfccs = np.mean(librosa.feature.mfcc(y=librosa.util.normalize(X), sr=sr, n_mfcc=13).T, axis=0)
        ext_features = np.expand_dims(mfccs, axis=0)

        logging.debug(f'{ext_features.shape}, {np.mean(ext_features)}, {np.std(ext_features)}')

        # classification
        pred = self.model.predict(ext_features)

        # logging.info(f'Pred: {pred}')

        return pred[0]
    
    def update_model_if_new(self):
        new_model_hash = hashlib.md5(open(MODEL, 'rb').read()).hexdigest()
        if new_model_hash != self.model_hash:
            logging.info('New model version detected! Updating model now.')
            self.model = pickle.load(open(MODEL, 'rb'))
            self.model_hash = new_model_hash
            logging.info(f'Loaded model with hash: {self.model_hash}')


    def run(self):
        logging.info('Staring Detection Thread!')
        loop_count = 0
        while True:
            logging.debug(f'Detection Q length: {self.bell_queue.qsize()}')
            sr, recording, recording_ts = self.bell_queue.get()
            start_detection = time()
            try:
                logging.debug(f'processing recording @ {recording_ts}')
                class_out = self.process_recording(recording, sr)
                logging.debug(f'{class_out}')
                datetimestr = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
                if class_out == 1:
                    logging.info('Bell detected!! Sending push notification to queue.')
                    self.notif_queue.put((datetimestr))
                    self.save_queue.put((recording, datetimestr))
            except Exception as e:
                logging.error(e)
            finally:
                self.bell_queue.task_done()
                if datetime.now() - self.last_updated > timedelta(minutes=self.model_check_interval):
                    self.update_model_if_new()
            # logging.info(f'Elapsed Detection Time: {time() - start_detection}')
            # logging.info(f'Queue size: {self.bell_queue.qsize()}')


def main():
    # Create a bell_queue to communicate with the worker threads
    bell_queue = Queue()
    # Create a notif_queue to send push notifications
    notif_queue = Queue()
    # Create a save_queue to save sounds for later reuse in training
    save_queue = Queue()

    # Start processing thread first
    det_worker = DetectionWorker(bell_queue, notif_queue, save_queue)
    # Setting daemon to True will let the main thread exit even though the workers are blocking
    # det_worker.daemon = False
    det_worker.start()

    # Start recording
    rec_worker = RecordingWorker(bell_queue, 0.5, SR)
    # Setting daemon to True will let the main thread exit even though the workers are blocking
    # rec_worker.daemon = False
    rec_worker.start()

    # Start the push notification listener
    notif_worker = PushWorker(notif_queue)
    # Setting daemon to True will let the main thread exit even though the workers are blocking
    # notif_worker.daemon = False
    notif_worker.start()

    # Start the saving worker which will save all bell instances
    save_worker = SaveWorker(save_queue, SR, location=SAVE_LOCATION)
    # Setting daemon to True will let the main thread exit even though the workers are blocking
    # save_worker.daemon = False
    save_worker.start()

    # Causes the main thread to wait for the bell_queue to finish processing all the tasks
    det_worker.join()
    rec_worker.join()
    notif_worker.join()
    save_worker.join()

if __name__ == '__main__':
    main()
