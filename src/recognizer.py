import queue, sys
import pyaudio
from vosk import Model, KaldiRecognizer

class SpeechRecognizer:
    def __init__(self, model_path: str, sample_rate: int = 16000):
        self.model = Model(model_path)
        self.recognizer = KaldiRecognizer(self.model, sample_rate)
        self.sample_rate = sample_rate
        self.q = queue.Queue()
        self.device_index = 0

    def _callback(self, in_data, *_args):
        self.q.put(in_data)
        return (None, pyaudio.paContinue)

    def listen(self):
        pa = pyaudio.PyAudio()
        stream = pa.open(format=pyaudio.paInt16,
                         channels=1,
                         rate=self.sample_rate,
                         input=True,
                         frames_per_buffer=8000,
                         stream_callback=self._callback,
                         input_device_index =self.device_index      )
        stream.start_stream()
        transcript = []
        try:
            while True:
                data = self.q.get()
                if self.recognizer.AcceptWaveform(data):
                    res = self.recognizer.Result()
                    transcript.append(eval(res)["text"])
        except KeyboardInterrupt:
            pass
        finally:
            stream.stop_stream(), stream.close(), pa.terminate()
        return " ".join(transcript).strip()
