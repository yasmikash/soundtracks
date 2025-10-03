import json
import os
from audiocraft.models import MusicGen
import torch
import soundfile as sf

# Load the prompts JSON file
print("Loading prompts.json...")
with open("prompts.json", "r") as f:
    prompts_data = json.load(f)

print(f"Loaded {len(prompts_data)} prompts from JSON file")

# Initialize the MusicGen model
print("Initializing MusicGen model...")
model = MusicGen.get_pretrained("facebook/musicgen-medium")
model.set_generation_params(duration=60)
print("Model initialized successfully")

# Create output directory if it doesn't exist
output_dir = "generated_tracks"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    print(f"Created output directory: {output_dir}")

# Process each prompt
for i, item in enumerate(prompts_data, 1):
    item_id = item["id"]
    prompt_text = item["prompt"]
    tone = item["tone"]
    keywords = item["keywords"]

    print(f"\n--- Processing item {i}/{len(prompts_data)} ---")
    print(f"ID: {item_id}")
    print(f"Tone: {tone}")
    print(f"Keywords: {', '.join(keywords)}")
    print(f"Prompt: {prompt_text[:100]}...")

    try:
        # Generate audio
        print("Generating audio...")
        audio = model.generate([prompt_text])

        # Convert to numpy array
        waveform = audio[0].cpu().numpy()
        waveform = waveform.T

        # Save the file
        filename = f"{item_id}.wav"
        filepath = os.path.join(output_dir, filename)
        sf.write(filepath, waveform, 32000)

        print(f"✅ Successfully saved: {filepath}")

    except Exception as e:
        print(f"❌ Error processing item {item_id}: {str(e)}")
        continue

print(
    f"\n🎵 Track generation complete! Generated {len(prompts_data)} tracks in '{output_dir}' directory"
)
