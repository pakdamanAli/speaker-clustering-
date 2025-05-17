import librosa
import torch
from speechbrain.pretrained import EncoderClassifier

SAMPLE_RATE = 16000
device = "cuda" if torch.cuda.is_available() else "cpu"

classifier = EncoderClassifier.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb",
    run_opts={"device": device},
    savedir="pretrained_models/spkrec-ecapa-voxceleb",
)


def extract_embedding(audio_path):
    try:
        waveform, sr = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)
        waveform = torch.tensor(waveform).unsqueeze(0).to(device)
        with torch.no_grad():
            embedding = classifier.encode_batch(waveform)
        return embedding.squeeze().cpu().numpy()
    except Exception as e:
        print(f"خطا در پردازش {audio_path}: {str(e)}")
        return None
