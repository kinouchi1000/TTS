import sys
import os
import string
import time
import argparse
import json
import numpy as np
import torch
import torchaudio
import pydub
import wave
import ffmpeg_normalize

from TTS.config import load_config
from TTS.utils.audio import AudioProcessor

from TTS.tts.utils.speakers import SpeakerManager
import librosa
from TTS.tts.models import setup_model
from TTS.tts.models.vits import Vits

"""
TTS CLIではYourTTS のVoice ConversionのInferenceの実装がされてなかったので、
スクリプトを組んで実装した。

https://github.com/coqui-ai/TTS/issues/1672

e.g. 20220928時点
python inference.py \
    --model_path exp_yourTTS_vctk_jvs/yourTTS_jvs_vctk-September-28-2022_07+49AM-9ae897d6/best_model.pth \
    --config_path exp_yourTTS_vctk_jvs/yourTTS_jvs_vctk-September-28-2022_07+49AM-9ae897d6/config.json \
    --SE_config_path download/config_se.json --SE_model_path download/SE_checkpoint.pth.tar \
    --target_path sample_wav/asano.wav \
    --source_path sample_wav/fukushima.wav \
    --tts_speaker exp/jvs_vctk_speakers.json \
    --output_path ./ \
    --language ja
"""


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str)
    parser.add_argument("--config_path", type=str)
    parser.add_argument("--SE_model_path", type=str)
    parser.add_argument("--SE_config_path", type=str)
    parser.add_argument("--target_path", type=str)
    parser.add_argument("--source_path", type=str)
    parser.add_argument("--tts_speaker", type=str)
    parser.add_argument("--output_path", type=str)
    parser.add_argument("--language", type=str)
    parser.add_argument("--use_cuda", action="store_true")
    return parser


def main():

    parser = get_parser()
    args = parser.parse_args()

    # path difinition
    model_path = args.model_path
    config_path = args.config_path
    se_model_path = args.SE_model_path
    se_config_path = args.SE_config_path
    target_path = args.target_path
    source_path = args.source_path
    tts_speaker = args.tts_speaker
    output_path = args.output_path
    language = args.language

    # load the config
    config = load_config(config_path)
    se_config = load_config(se_config_path)

    # load the audio processor
    ap = AudioProcessor(**config.audio)

    config.model_args["d_vector_file"] = tts_speaker
    config.model_args["use_speaker_encoder_as_loss"] = False
    config.model_args["use_d_vector_file"] = True

    # model 構築
    model: Vits = setup_model(config)
    cp = torch.load(model_path, map_location=torch.device("cpu"))
    model_weights = cp["model"].copy()
    for key in list(model_weights.keys()):
        if "speaker_encoder" in key:
            del model_weights[key]
    model.load_state_dict(model_weights, strict=False)
    model.eval()

    # CUDA 設定
    if args.use_cuda:
        model = model.cuda()

    # speaker encoder
    SE_speaker_manager = SpeakerManager(
        encoder_model_path=se_model_path, encoder_config_path=se_config_path, use_cuda=args.use_cuda
    )

    # voice conversion
    normalize = ffmpeg_normalize.FFmpegNormalize(sample_rate=16000, normalization_type="rms", target_level=-27)
    normalize.add_media_file(target_path, target_path)
    normalize.add_media_file(source_path, source_path)

    target_emb = SE_speaker_manager.compute_embedding_from_clip(target_path)
    target_emb = torch.FloatTensor(target_emb).unsqueeze(0)

    driving_emb = SE_speaker_manager.compute_embedding_from_clip(source_path)
    driving_emb = torch.FloatTensor(driving_emb).unsqueeze(0)

    driving_spec, sr = librosa.load(source_path, sr=ap.sample_rate)
    driving_spec = torch.FloatTensor(driving_spec).unsqueeze(0)  # [B, T]

    if args.use_cuda:
        ref_wav_voc = model.inference_voice_conversion(
            reference_wav=driving_spec.cuda(), d_vector=driving_emb.cuda(), reference_d_vector=target_emb.cuda()
        )
        ref_wav_voc = ref_wav_voc.squeeze().cpu().detach()
    else:
        ref_wav_voc = model.inference_voice_conversion(
            reference_wav=driving_spec, d_vector=driving_emb, reference_d_vector=target_emb
        )
        ref_wav_voc: torch.Tensor = ref_wav_voc.squeeze().detach()
    print(ref_wav_voc.shape)

    # 音声を保存
    ref_wav_voc = ref_wav_voc.unsqueeze(0)  # [C, W]
    print(ref_wav_voc.shape)
    torchaudio.save(filepath=os.path.join(output_path, "vc_sound.wav"), src=ref_wav_voc, sample_rate=sr)


if __name__ == "__main__":
    main()
