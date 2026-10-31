# model_loader.py

import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor


class RecognitionModel:

    def __init__(self):
        self.model = self.load_model()
        self.processor = self.load_processor()
        self.id2phoneme = {
            1: "aa", 2: "ae", 3: "ah", 4: "aw", 5: "ay", 6: "b", 7: "ch", 8: "d", 9: "dh",
            10: "dx", 11: "eh", 12: "er", 13: "ey", 14: "f", 15: "g", 16: "h#", 17: "hh",
            18: "ih", 19: "iy", 20: "jh", 21: "k", 22: "l", 23: "m", 24: "n", 25: "ng",
            26: "ow", 27: "oy", 28: "p", 29: "r", 30: "s", 31: "sh", 32: "t", 33: "th",
            34: "uh", 35: "uw", 36: "v", 37: "w", 38: "y", 39: "z", 0: "|", 40: "[UNK]", 41: "[PAD]"
        }


    def load_model(self):
        """
        Load the Wav2Vec2 model.
        """
        model = Wav2Vec2ForCTC.from_pretrained("./model/")
        model.eval()
        print("Model and processor loaded successfully.")
        return model


    def load_processor(self):
        """
        Load the Wav2Vec2 processor.
        """
        processor = Wav2Vec2Processor.from_pretrained("./model/")
        print("Processor loaded successfully.")
        return processor


    def predict(self, audio, sampling_rate=16000):
        """ Predict phonemes from audio input.
        This method takes an audio input and returns the predicted phonemes.

        Args:
            audio (np.array): An array representing the audio waveform.
            sampling_rate (int): Sampling rate of the waveform. Defaults to 16000.

        Returns:
            _type_: _description_
        """
        input_values = self.processor(audio, sampling_rate=sampling_rate, return_tensors="pt").input_values
        with torch.no_grad():
            logits = self.model(input_values).logits
        pred_ids = torch.argmax(logits, dim=-1)[0]
        phonemes = [self.id2phoneme.get(int(i), "[UNK]") for i in pred_ids if int(i) not in [0, 41]]
        return phonemes