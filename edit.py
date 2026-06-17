import os
import argparse
import random
import base64
import numpy as np
import torch
from tqdm import tqdm
from scipy.io.wavfile import write

import utils
from models import SynthesizerTrn, SpeakerFlow
from ecapa import ECAPA_TDNN
from mel_processing import spectrogram_torch
from utils import load_wav_to_torch
from text.symbols import symbols



def save_wav_int16(path, sr, audio_float):
    audio = np.asarray(audio_float, dtype=np.float32)
    audio = np.nan_to_num(audio)
    audio_int16 = (audio * 32768.0).astype(np.int16)
    write(path, sr, audio_int16)


def audio_to_spectrogram(filename, hps):
    if os.path.exists(filename.replace(".wav", "_trimmed.wav")):
        filename = filename.replace(".wav", "_trimmed.wav")

    audio, sampling_rate = load_wav_to_torch(filename)
    assert sampling_rate == hps.data.sampling_rate, \
        f"Sample rate mismatch: {sampling_rate} vs {hps.data.sampling_rate}"

    spec_filename = filename.replace(".wav", ".spec.pt")
    if os.path.exists(spec_filename):
        spec = torch.load(spec_filename)
    else:
        audio_norm = audio / hps.data.max_wav_value
        audio_norm = audio_norm.unsqueeze(0)
        spec = spectrogram_torch(
            audio_norm,
            hps.data.filter_length,
            hps.data.sampling_rate,
            hps.data.hop_length,
            hps.data.win_length,
            center=False
        )
        spec = torch.squeeze(spec, 0)

    return spec


def edit_voice(model, speaker_flow, ecapa, hps, audio_path, source_age, source_sex,
               target_age=None, target_sex=None, device='cuda'):
    """Edit voice age or gender via SpeakerFlow transformation."""
    spec = audio_to_spectrogram(audio_path, hps).unsqueeze(0).to(device)
    spec_len = torch.LongTensor([spec.shape[-1]]).to(device)

    final_age = target_age if target_age is not None else source_age
    final_sex = target_sex if target_sex is not None else source_sex

    # Decade-based age bins: 20s=1, 30s=2, ..., 80s=7
    source_age_bin = (int(source_age) // 10) - 1
    final_age_bin = (int(final_age) // 10) - 1

    age_src = torch.LongTensor([source_age_bin]).to(device)
    sex_src = torch.LongTensor([source_sex]).to(device)
    age_tgt = torch.LongTensor([final_age_bin]).to(device)
    sex_tgt = torch.LongTensor([final_sex]).to(device)

    print(f"  Age: {source_age} -> {final_age}, Sex: {source_sex} -> {final_sex}")

    with torch.no_grad():
        spec_t = torch.transpose(spec, 1, 2)
        g_src = ecapa.get_embed(spec_t, lengths=spec_len).unsqueeze(-1)

        g_z, _ = speaker_flow(g_src, age_src, sex_src)
        g_tgt = speaker_flow(g_z, age_tgt, sex_tgt, reverse=True)

        edited_audio, y_mask, (z, z_p, z_hat) = model.voice_conversion(
            spec, spec_len, g_src, g_tgt
        )

    return edited_audio.squeeze(0).squeeze(0).cpu().numpy()


def audio_to_base64(audio_path):
    try:
        with open(audio_path, 'rb') as f:
            return base64.b64encode(f.read()).decode('utf-8')
    except Exception as e:
        print(f"Error converting {audio_path}: {e}")
        return ""


def create_audio_element(audio_path, label="Audio"):
    if not os.path.exists(audio_path):
        return '<div class="audio-container"><span>File not found</span></div>'

    base64_data = audio_to_base64(audio_path)
    if not base64_data:
        return '<div class="audio-container"><span>Error loading audio</span></div>'

    data_url = f"data:audio/wav;base64,{base64_data}"
    return f'''
    <div class="audio-container">
        <audio controls>
            <source src="{data_url}" type="audio/wav">
            Your browser does not support the audio element.
        </audio>
        <div class="audio-label">{label}</div>
    </div>
    '''


def load_samples(data_file, num_samples=20):
    """Load and randomly select samples from a filelist."""
    print(f"Loading samples from {data_file}...")

    with open(data_file, 'r', encoding='utf-8') as f:
        lines = f.read().splitlines()

    samples = []
    for line in lines:
        parts = line.split("|")
        if len(parts) >= 4:
            try:
                audio_path = parts[0]
                age = int(parts[2])
                sex = int(parts[3])
                if os.path.exists(audio_path):
                    samples.append((audio_path, age, sex))
            except Exception:
                continue

    print(f"Found {len(samples)} valid samples")

    if len(samples) > num_samples:
        random.seed(42)
        samples = random.sample(samples, num_samples)

    print(f"Selected {len(samples)} samples for generation")
    return samples


def generate_edited_samples(model, speaker_flow, ecapa, hps, samples, output_dir, device='cuda'):
    """Generate age and gender edits for each sample."""
    os.makedirs(output_dir, exist_ok=True)
    print(f"\nGenerating edited samples -> {output_dir}\n")

    total_generated = 0
    html_data = []

    for i, (audio_path, age, sex) in enumerate(tqdm(samples, desc="Processing")):
        basename = os.path.splitext(os.path.basename(audio_path))[0]
        print(f"\n[{i+1}/{len(samples)}] {basename} | age={age}, sex={'F' if sex == 1 else 'M'}")

        # Age edit
        if age < 40:
            target_age, transformation_type = 50, "young_to_old"
        else:
            target_age, transformation_type = 20, "old_to_young"

        edited_audio = edit_voice(model, speaker_flow, ecapa, hps, audio_path,
                                  age, sex, target_age=target_age, device=device)
        age_output_path = os.path.join(output_dir, f"{basename}_age{age}to{target_age}.wav")
        save_wav_int16(age_output_path, hps.data.sampling_rate, edited_audio)

        with open(age_output_path.replace('.wav', '.meta.txt'), 'w') as f:
            f.write(f"transformation_type={transformation_type}\n"
                    f"source_age={age}\ntarget_age={target_age}\n"
                    f"source_sex={sex}\ntarget_sex={sex}\nsource_audio={audio_path}\n")

        total_generated += 1
        print(f"  Saved age edit: {os.path.basename(age_output_path)}")

        # Gender edit
        target_sex = 0 if sex == 1 else 1
        sex_str = "F" if sex == 1 else "M"
        target_sex_str = "F" if target_sex == 1 else "M"

        edited_audio = edit_voice(model, speaker_flow, ecapa, hps, audio_path,
                                  age, sex, target_sex=target_sex, device=device)
        gender_output_path = os.path.join(output_dir, f"{basename}_sex{sex_str}to{target_sex_str}.wav")
        save_wav_int16(gender_output_path, hps.data.sampling_rate, edited_audio)

        with open(gender_output_path.replace('.wav', '.meta.txt'), 'w') as f:
            f.write(f"transformation_type=gender_change\n"
                    f"source_age={age}\ntarget_age={age}\n"
                    f"source_sex={sex}\ntarget_sex={target_sex}\nsource_audio={audio_path}\n")

        total_generated += 1
        print(f"  Saved gender edit: {os.path.basename(gender_output_path)}")

        html_data.append({
            'basename': basename,
            'source_audio': audio_path,
            'source_age': age,
            'source_sex': sex,
            'target_age': target_age,
            'age_transformation': transformation_type,
            'age_output': age_output_path,
            'target_sex': target_sex,
            'gender_output': gender_output_path,
        })

    print(f"\nGeneration complete. {total_generated} files written to {output_dir}\n")
    return html_data


def create_html_demo(html_data, output_html_path):
    """Create a self-contained HTML demo page with embedded audio."""
    print("Creating HTML demo page...")

    table_rows = ""
    for i, data in enumerate(html_data):
        sex_str = "Female" if data['source_sex'] == 1 else "Male"
        target_sex_str = "Female" if data['target_sex'] == 1 else "Male"
        age_desc = "Young to Old" if data['age_transformation'] == "young_to_old" else "Old to Young"
        gender_desc = f"{sex_str} to {target_sex_str}"

        table_rows += f'''
        <tr>
            <td>{i+1}</td>
            <td>
                <div class="speaker-info">Age: {data['source_age']}</div>
                <div class="speaker-info">Gender: {sex_str}</div>
                <div class="sample-id">{data['basename']}</div>
                {create_audio_element(data['source_audio'], "Original")}
            </td>
            <td>
                <div class="speaker-info">Age: {data['source_age']} to {data['target_age']}</div>
                <div class="speaker-info">Gender: {sex_str}</div>
                <div class="conversion-type">{age_desc}</div>
                {create_audio_element(data['age_output'], "Age Edit")}
            </td>
            <td>
                <div class="speaker-info">Age: {data['source_age']}</div>
                <div class="speaker-info">Gender: {gender_desc}</div>
                <div class="conversion-type">Gender Change</div>
                {create_audio_element(data['gender_output'], "Gender Edit")}
            </td>
        </tr>
        '''

    html_content = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>RIVET: Robust Idempotent Voice Attribute Editing</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background: #f5f5f5;
            min-height: 100vh;
            color: #333;
        }}
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            background: #fff;
            border-radius: 8px;
            padding: 40px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }}
        h1 {{
            text-align: center;
            color: #C41230;
            margin-bottom: 6px;
            font-size: 2em;
            font-weight: 700;
        }}
        .subtitle {{
            text-align: center;
            color: #555;
            margin-bottom: 30px;
            font-size: 1.1em;
        }}
        .info-box {{
            background: #fafafa;
            border: 1px solid #e0e0e0;
            border-left: 4px solid #C41230;
            padding: 16px 20px;
            border-radius: 4px;
            margin-bottom: 28px;
        }}
        .info-box h2 {{ margin: 0 0 8px; font-size: 1.1em; color: #C41230; }}
        .info-box p {{ margin: 4px 0; font-size: 0.95em; color: #444; }}
        .demo-table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 20px;
            font-size: 0.9em;
        }}
        .demo-table th {{
            background: #C41230;
            color: white;
            padding: 12px 10px;
            text-align: center;
        }}
        .demo-table td {{
            padding: 14px 10px;
            text-align: center;
            border-bottom: 1px solid #eee;
            vertical-align: middle;
        }}
        .demo-table tbody tr:hover {{ background: #fafafa; }}
        .speaker-info {{ font-size: 0.85em; color: #444; margin-bottom: 4px; }}
        .sample-id {{ font-size: 0.75em; color: #aaa; margin-bottom: 8px; font-family: monospace; }}
        .audio-container {{ display: flex; flex-direction: column; align-items: center; gap: 4px; margin-top: 8px; }}
        .audio-label {{ font-size: 0.78em; color: #888; }}
        audio {{ width: 200px; height: 36px; }}
        .conversion-type {{
            font-size: 0.82em;
            color: #555;
            padding: 4px 8px;
            background: #f0f0f0;
            border-radius: 4px;
            margin-bottom: 6px;
        }}
        .stats {{
            display: flex;
            justify-content: space-around;
            margin-top: 30px;
            padding: 16px;
            background: #fafafa;
            border-radius: 8px;
            border: 1px solid #e0e0e0;
        }}
        .stat-item {{ text-align: center; }}
        .stat-value {{ font-size: 1.8em; font-weight: 700; color: #C41230; }}
        .stat-label {{ font-size: 0.9em; color: #777; margin-top: 4px; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>RIVET: Robust Idempotent Voice Attribute Editing</h1>
        <p class="subtitle">Audio samples — age and gender transformation</p>

        <div class="info-box">
            <h2>Model</h2>
            <p><strong>ECAPA-TDNN</strong> extracts a fixed-length speaker embedding from input speech.</p>
            <p><strong>SpeakerFlow</strong> (conditional normalizing flow) maps the embedding to the target attribute space.</p>
            <p><strong>VITS generator</strong> reconstructs speech conditioned on the transformed embedding.</p>
        </div>

        <table class="demo-table">
            <thead>
                <tr>
                    <th>#</th>
                    <th>Original</th>
                    <th>Age Edit</th>
                    <th>Gender Edit</th>
                </tr>
            </thead>
            <tbody>
{table_rows}
            </tbody>
        </table>

        <div class="stats">
            <div class="stat-item">
                <div class="stat-value">{len(html_data)}</div>
                <div class="stat-label">Samples</div>
            </div>
            <div class="stat-item">
                <div class="stat-value">{len(html_data) * 2}</div>
                <div class="stat-label">Edits Generated</div>
            </div>
            <div class="stat-item">
                <div class="stat-value">2</div>
                <div class="stat-label">Edit Types</div>
            </div>
        </div>
    </div>
</body>
</html>
'''

    with open(output_html_path, 'w', encoding='utf-8') as f:
        f.write(html_content)

    print(f"HTML demo saved to: {output_html_path}")


def load_models(checkpoint_dir, checkpoint_step, hps, device):
    """Load ECAPA, generator, and SpeakerFlow from a checkpoint directory."""
    spec_channels = hps.data.filter_length // 2 + 1

    net_ecapa = ECAPA_TDNN(input_size=spec_channels).to(device)
    net_ecapa.eval()
    utils.load_checkpoint(
        os.path.join(checkpoint_dir, f'ECAPA_{checkpoint_step}.pth'),
        net_ecapa, None
    )

    net_g = SynthesizerTrn(
        len(symbols), spec_channels,
        hps.train.segment_size // hps.data.hop_length,
        n_speakers=hps.data.n_speakers,
        **hps.model
    ).to(device)
    net_g.eval()
    utils.load_checkpoint(
        os.path.join(checkpoint_dir, f'G_{checkpoint_step}.pth'),
        net_g, None
    )

    speaker_flow = SpeakerFlow(512, 512, 5, 1, 4, gin_channels=512).to(device)
    speaker_flow.eval()
    checkpoint = torch.load(
        os.path.join(checkpoint_dir, f'CNF_{checkpoint_step}.pth'),
        map_location=device
    )
    speaker_flow.load_state_dict(checkpoint['model'])

    return net_ecapa, net_g, speaker_flow


def main():
    parser = argparse.ArgumentParser(description='Generate RIVET voice edits (age/gender)')
    parser.add_argument('--config', type=str, default='configs/globe_base.json')
    parser.add_argument('--checkpoint_dir', type=str, default='./logs/ours')
    parser.add_argument('--checkpoint_step', type=int, default=71000)
    parser.add_argument('--data_file', type=str, required=True,
                        help='Filelist in format: audio_path|sid|age|gender|text')
    parser.add_argument('--output_dir', type=str, default='edited_samples')
    parser.add_argument('--num_samples', type=int, default=20)
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    print(f"Config: {args.config}")
    print(f"Checkpoint: {args.checkpoint_dir} @ step {args.checkpoint_step}")
    print(f"Output: {args.output_dir}\n")

    hps = utils.get_hparams_from_file(args.config)
    net_ecapa, net_g, speaker_flow = load_models(
        args.checkpoint_dir, args.checkpoint_step, hps, args.device
    )
    print("Models loaded.\n")

    samples = load_samples(args.data_file, args.num_samples)
    html_data = generate_edited_samples(
        net_g, speaker_flow, net_ecapa, hps, samples, args.output_dir, args.device
    )

    html_output_path = os.path.join(args.output_dir, 'demo.html')
    create_html_demo(html_data, html_output_path)
    print(f"\nDone. Audio: {args.output_dir} | Demo: {html_output_path}")


if __name__ == "__main__":
    main()
