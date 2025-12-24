from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse
from transformers import (
    WhisperForConditionalGeneration,
    WhisperProcessor,
    AutoTokenizer,
    AutoModelForSequenceClassification
)
import torch
import torch.nn.functional as F
import torchaudio
import io

# ================= INIT =================
app = FastAPI(title="Speech Emotion Demo")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ===== WHISPER =====
WHISPER_DIR = "whisper_merged"

whisper_processor = WhisperProcessor.from_pretrained(WHISPER_DIR)
whisper_model = WhisperForConditionalGeneration.from_pretrained(WHISPER_DIR)
whisper_model.to(device)
whisper_model.eval()

# ===== PHOBERT =====
PHOBERT_DIR = "Phobert_ft1"

phobert_tokenizer = AutoTokenizer.from_pretrained(PHOBERT_DIR)
phobert_model = AutoModelForSequenceClassification.from_pretrained(PHOBERT_DIR)
phobert_model.to(device)
phobert_model.eval()

labels_map_vi = {
    0: "Giận dữ",
    1: "Ghê tởm",
    2: "Sợ hãi",
    3: "Vui vẻ",
    4: "Buồn bã",
    5: "Ngạc nhiên",
    6: "Khác"
}

# ================= UTILS =================
def whisper_transcribe(waveform, sr):
    if sr != 16000:
        waveform = torchaudio.functional.resample(waveform, sr, 16000)

    if waveform.ndim > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    waveform = waveform.squeeze().numpy()

    inputs = whisper_processor.feature_extractor(
        waveform,
        sampling_rate=16000,
        return_tensors="pt"
    ).input_features.to(device)

    with torch.no_grad():
        ids = whisper_model.generate(inputs, max_length=225)

    text = whisper_processor.tokenizer.decode(
        ids[0],
        skip_special_tokens=True
    )
    return text.strip().lower()


def predict_emotion(text):
    if len(text.split()) < 2:
        return "Khác", 1.0

    encoding = phobert_tokenizer(
        text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=128
    )
    encoding = {k: v.to(device) for k, v in encoding.items()}

    with torch.no_grad():
        logits = phobert_model(**encoding).logits

    probs = F.softmax(logits, dim=-1)
    conf, pred = torch.max(probs, dim=-1)

    return labels_map_vi[pred.item()], float(conf.item())


# ================= API =================
@app.post("/analyze")
async def analyze_audio(file: UploadFile = File(...)):
    audio_bytes = await file.read()
    waveform, sr = torchaudio.load(io.BytesIO(audio_bytes))

    text = whisper_transcribe(waveform, sr)
    emotion, confidence = predict_emotion(text)

    return {
        "text": text,
        "emotion": emotion,
        "confidence": confidence
    }


# ================= WEB UI =================
@app.get("/", response_class=HTMLResponse)
def home():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Speech Emotion Demo</title>
        <style>
            body {
                font-family: Arial;
                background: #f4f6f8;
                display: flex;
                justify-content: center;
                margin-top: 40px;
            }
            .card {
                background: white;
                padding: 30px;
                border-radius: 12px;
                width: 420px;
                box-shadow: 0 4px 12px rgba(0,0,0,0.1);
            }
            h2 { text-align: center; }
            button {
                width: 100%;
                padding: 10px;
                background: #2563eb;
                color: white;
                border: none;
                border-radius: 8px;
                cursor: pointer;
            }
            .result {
                margin-top: 20px;
                background: #eef2ff;
                padding: 10px;
                border-radius: 8px;
            }
        </style>
    </head>
    <body>
        <div class="card">
            <h2>🎙️ Phân tích cảm xúc giọng nói</h2>
            <input type="file" id="audio" />
            <br><br>
            <button onclick="send()">Phân tích</button>
            <div class="result" id="result"></div>
        </div>

        <script>
            async function send() {
                const fileInput = document.getElementById("audio");
                if (!fileInput.files.length) {
                    alert("Vui lòng chọn file audio!");
                    return;
                }

                const formData = new FormData();
                formData.append("file", fileInput.files[0]);

                const res = await fetch("/analyze", {
                    method: "POST",
                    body: formData
                });

                const data = await res.json();

                document.getElementById("result").innerHTML = `
                    <b>Văn bản:</b> ${data.text}<br>
                    <b>Cảm xúc:</b> ${data.emotion}<br>
                    <b>Độ tin cậy:</b> ${data.confidence}
                `;
            }
        </script>
    </body>
    </html>
    """
