import os
from glob import glob
import sys
from trainer import Trainer, TrainerArgs
import pprint
from TTS.tts.configs.shared_configs import BaseDatasetConfig
from TTS.tts.configs.vits_config import VitsConfig
from TTS.tts.datasets import load_tts_samples
from TTS.tts.models.vits import CharactersConfig, Vits, VitsArgs, VitsAudioConfig
from TTS.tts.utils.languages import LanguageManager
from TTS.tts.utils.speakers import SpeakerManager
from TTS.tts.utils.text.tokenizer import TTSTokenizer
from TTS.utils.audio import AudioProcessor
from TTS.config import load_config
from TTS.tts.models import setup_model
import torch


output_path = "out/"
config_path = "download/config.json"
model_path = "download/best_model.pth.tar"
speaker_path = "download/speaker.json"

dataset_path = "../jvs_ver1"
dataset_config = [BaseDatasetConfig(formatter="jvs", meta_file_train=None, path=dataset_path, language="ja")]

config = load_config(config_path)
# confirm json formmat
config.from_dict(config.to_dict())

config["characters"]["is_unique"] = False
config["datasets"] = dataset_config
config["model_args"]["d_vector_file"] = speaker_path
pprint.pprint(config.to_dict())
# init audio processor
# ap = AudioProcessor(**config.audio.to_dict())

# load training samples
train_samples, eval_samples = load_tts_samples(
    config["datasets"],
    eval_split=True,
    eval_split_max_size=config.eval_split_max_size,
    eval_split_size=config.eval_split_size,
)

# init speaker manager for multi-speaker training
# it maps speaker-id to speaker-name in the model and data-loader
# speaker_manager = SpeakerManager()
# speaker_manager.set_ids_from_data(train_samples + eval_samples, parse_key="speaker_name")
# config.model_args.num_speakers = speaker_manager.num_speakers

# language_manager = LanguageManager(config=config)
# config.model_args.num_languages = language_manager.num_languages

# INITIALIZE THE TOKENIZER
# Tokenizer is used to convert text to sequences of token IDs.
# config is updated with the default characters if not defined in the config.
# tokenizer, config = TTSTokenizer.init_from_config(config)

# init model
model = setup_model(config)  # Vits(config, ap, tokenizer, speaker_manager, language_manager)

# load parameters
cp = torch.load(model_path, map_location=torch.device("cpu"))
model_weights = cp["model"].copy()
for key in list(model_weights.keys()):
    if "speaker_encoder" in key:
        del model_weights[key]

model.load_state_dict(model_weights)
model = model.cuda()

# init the trainer and 🚀
trainer = Trainer(
    TrainerArgs(), config, output_path, model=model, train_samples=train_samples, eval_samples=eval_samples
)
trainer.fit()
