# Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.
# MIT licence – see original header for details.

import argparse, json, datetime as dt
from pathlib import Path
import numpy as np
from scipy.io.wavfile import write
import torch

import params
from model import GradTTS
from text import text_to_sequence, cmudict
from text.symbols import symbols
from utils import intersperse

import sys
sys.path.append('./hifi-gan/')
from env import AttrDict
from models import Generator as HiFiGAN

# --------------------------------------------------------------------------- #
# defaults (identical to upstream)
DEFAULT_HIFI_CFG  = None # './checkpts/hifigan-config.json'
DEFAULT_HIFI_CKPT = None # './checkpts/hifigan.pt'
# --------------------------------------------------------------------------- #

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--file', required=True,
                        help='Text‑file with one sentence per line')
    parser.add_argument('-c', '--checkpoint', required=True,
                        help='Grad‑TTS checkpoint (*.pt)')
    parser.add_argument('-t', '--timesteps', type=int, default=10,
                        help='# reverse‑diffusion steps (lower = faster, lower quality)')
    parser.add_argument('-s', '--speaker_id', type=int, default=None,
                        help='Speaker ID for multi‑speaker models')
    # NEW -------------------------------------------------------------------- #
    parser.add_argument('--hifigan_cfg',  default=DEFAULT_HIFI_CFG,
                        help='HiFi‑GAN JSON config')
    parser.add_argument('--hifigan_ckpt', default=DEFAULT_HIFI_CKPT,
                        help='HiFi‑GAN generator checkpoint (*.pt)')
    parser.add_argument('-o', '--output_dir', default='./out',
                        help='Directory to save synthesized wavs')
    # ----------------------------------------------------------------------- #
    args = parser.parse_args()

    # ----------------------------- Grad‑TTS --------------------------------- #
    spk = (torch.LongTensor([args.speaker_id]).cuda()
           if args.speaker_id is not None else None)

    print('Initializing Grad‑TTS …')
    generator = GradTTS(len(symbols) + 1, params.n_spks, params.spk_emb_dim,
                        params.n_enc_channels, params.filter_channels,
                        params.filter_channels_dp, params.n_heads, params.n_enc_layers,
                        params.enc_kernel, params.enc_dropout, params.window_size,
                        params.n_feats, params.dec_dim,
                        params.beta_min, params.beta_max, params.pe_scale)
    generator.load_state_dict(
        torch.load(args.checkpoint, map_location='cpu'))
    _ = generator.cuda().eval()
    print(f'Grad‑TTS parameters: {generator.nparams/1e6:.2f} M')

    # ----------------------------- HiFi‑GAN -------------------------------- #
    print('Initializing HiFi‑GAN …')
    with open(args.hifigan_cfg) as f:
        h = AttrDict(json.load(f))
    vocoder = HiFiGAN(h)
    vocoder.load_state_dict(
        torch.load(args.hifigan_ckpt, map_location='cpu')['generator'])
    _ = vocoder.cuda().eval()
    vocoder.remove_weight_norm()

    # ----------------------------- I/O ------------------------------------- #
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    texts = Path(args.file).read_text(encoding='utf‑8').splitlines()
    cmu   = cmudict.CMUDict('./resources/cmu_dictionary')

    with torch.no_grad():
        for idx, text in enumerate(texts):
            if not text.strip():
                continue  # skip blank lines
            print(f'[{idx}] synthesizing: “{text[:60]}…”')

            x = torch.LongTensor(
                intersperse(text_to_sequence(text, dictionary=cmu), len(symbols))
            ).unsqueeze(0).cuda()
            x_lens = torch.LongTensor([x.shape[-1]]).cuda()

            t0 = dt.datetime.now()
            _enc, mel, _attn = generator(
                x, x_lens, n_timesteps=args.timesteps,
                temperature=1.5, stoc=False, spk=spk, length_scale=0.91)
            rtf = (dt.datetime.now() - t0).total_seconds() * 22050 / (mel.shape[-1]*256)
            print(f'  • Grad‑TTS RTF ≈ {rtf:.3f}')

            wav = (vocoder(mel)
                   .cpu().squeeze().clamp(-1, 1).numpy() * 32768).astype(np.int16)
            write(out_dir / f'sample_{idx:03d}.wav', 22050, wav)

    print(f'Done → {out_dir.resolve()}')
