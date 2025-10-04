from stable_audio_tools import get_pretrained_model
from stable_audio_tools.inference.generation import generate_diffusion_cond
import torch
from einops import rearrange

device = "cuda" if torch.cuda.is_available() else "cpu"

model, model_config = get_pretrained_model("stabilityai/stable-audio-open-1.0")
model = model.to(device)

conditioning = [
    {
        "prompt": "An uplifting and inspiring background track with soaring strings, soft brass, and motivating rhythms. The atmosphere feels empowering and hopeful, with clear progression and emotional crescendos.",
        "seconds_start": 0,
        "seconds_total": 60,
    }
]

output = generate_diffusion_cond(
    model,
    steps=100,
    cfg_scale=7,
    conditioning=conditioning,
    sample_size=model_config["sample_size"],
    sigma_min=0.3,
    sigma_max=500,
    sampler_type="dpmpp-3m-sde",
    device=device,
)

output = rearrange(output, "b d n -> d (b n)")
# Normalize and save
import torchaudio

output = (
    output.to(torch.float32)
    .div(torch.max(torch.abs(output)))
    .clamp(-1, 1)
    .mul(32767)
    .to(torch.int16)
    .cpu()
)
torchaudio.save("./out/out.wav", output, model_config["sample_rate"])
