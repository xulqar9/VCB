"""Emotion in the user's voice: emotion2vec+ through FunASR (installed by
the launcher's voice emotion pack). Nine classes; only a confident,
non-neutral result is passed on."""

import logging

import numpy as np

log = logging.getLogger("vcb.vocal.emotion")

# emotion2vec+ class -> how the assistant is told about it.
EMOTION_PHRASES = {
    "angry": "sounds angry",
    "disgusted": "sounds disgusted",
    "fearful": "sounds scared or anxious",
    "happy": "sounds happy",
    "sad": "sounds sad",
    "surprised": "sounds surprised",
}


class EmotionRecognizer:
    def __init__(self, model_id: str, device: str) -> None:
        self.model_id = model_id
        self.device = device
        self._model = None

    def load(self) -> None:
        from funasr import AutoModel

        self._model = AutoModel(model=self.model_id, hub="hf", device=self.device,
                                disable_update=True, disable_pbar=True, disable_log=True)

    def classify(self, audio: np.ndarray) -> tuple[str, float]:
        """(emotion, probability) for a 16 kHz clip, e.g. ("happy", 0.83)."""
        result = self._model.generate(np.asarray(audio, np.float32), granularity="utterance",
                                      extract_embedding=False, disable_pbar=True)[0]
        # Labels are bilingual ("开心/happy"); scores follow the same order.
        scores = {label.split("/")[-1]: float(s)
                  for label, s in zip(result["labels"], result["scores"])}
        return max(scores.items(), key=lambda kv: kv[1])
