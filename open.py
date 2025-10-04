from stable_audio_tools import get_pretrained_model
from stable_audio_tools.inference.generation import generate_diffusion_cond
import torch
from einops import rearrange
import torchaudio
import os

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load model
model, model_config = get_pretrained_model("stabilityai/stable-audio-open-1.0")
model = model.to(device)

# Create output directory if needed
os.makedirs("./out", exist_ok=True)

# ---- CONFIGURATION ----
prompt = (
    "soft corporate background music with warm piano and airy synths, "
    "gradually adds gentle percussion and bass after 15 seconds, evolving but smooth, "
    "no vocals, clean mix, for voiceover commercial"
)

conditioning = [
    {
        "prompt": prompt,
        "seconds_start": 0,
        "seconds_total": 30,  # Length of clip (20–40 s works well)
    }
]

# ---- GENERATION ----
output = generate_diffusion_cond(
    model,
    steps=60,  # Faster & smoother than 100
    cfg_scale=5.5,  # Balanced adherence to prompt
    conditioning=conditioning,
    sample_size=model_config["sample_size"],
    sigma_min=0.3,
    sigma_max=400,  # Slightly reduced for smoother tone
    sampler_type="dpmpp-3m-sde",
    device=device,
)

# ---- POST-PROCESSING ----
output = rearrange(output, "b d n -> d (b n)")

output = (
    output.to(torch.float32)
    .div(torch.max(torch.abs(output)))  # normalize
    .clamp(-1, 1)
    .mul(32767)
    .to(torch.int16)
    .cpu()
)

# Optional: reduce gain slightly (2–3 dB lower for voice layering)
output = (output * 0.7).to(torch.int16)

# ---- SAVE ----
torchaudio.save("./out/out.wav", output, model_config["sample_rate"])
print("✅ Generated and saved: ./out/out.wav")
