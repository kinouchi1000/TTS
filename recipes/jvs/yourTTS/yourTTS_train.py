from email.policy import strict
from enum import unique
from trainer import Trainer, TrainerArgs
import pprint
from TTS.tts.configs.shared_configs import BaseDatasetConfig
from TTS.tts.configs.vits_config import VitsConfig
from TTS.tts.models.vits import VitsArgs
from TTS.tts.datasets import load_tts_samples

from TTS.tts.utils.text.tokenizer import TTSTokenizer
from TTS.utils.audio import AudioProcessor
from TTS.tts.utils.speakers import SpeakerManager
from TTS.tts.models.vits import Vits
from TTS.tts.configs.shared_configs import CharactersConfig
import wandb
import os

from TTS.config.shared_configs import BaseAudioConfig
import torch

output_path = "exp_yourTTS_ja"

dataset_config = [
    BaseDatasetConfig(
        formatter="jvs",
        dataset_name="jvs",
        meta_file_train="train",
        meta_file_val="dev",
        path="../jvs_ver1",
        language="ja",
        ignored_speakers=None,
    )
]

audio_config = BaseAudioConfig(
    sample_rate=16000,
    win_length=1024,
    hop_length=256,
    num_mels=80,
    preemphasis=0.0,
    ref_level_db=20,
    log_func="np.log",
    do_trim_silence=False,
    trim_db=23.0,
    mel_fmin=0,
    mel_fmax=None,
    spec_gain=1.0,
    signal_norm=True,
    do_amp_to_db_linear=False,
    resample=True,
)


vitsArgs = VitsArgs(
    use_language_embedding=False,
    use_speaker_embedding=True,
    use_sdp=False,
)

config = VitsConfig(
    model_args=vitsArgs,
    audio=audio_config,
    run_name="vits_mailabs",
    use_speaker_embedding=True,
    batch_size=32,
    eval_batch_size=16,
    batch_group_size=0,
    num_loader_workers=8,
    num_eval_loader_workers=8,
    run_eval=True,
    test_delay_epochs=-1,
    epochs=1000,
    text_cleaner="basic_german_cleaners",
    use_phonemes=False,
    phoneme_language="jp-ja",
    phoneme_cache_path=os.path.join(output_path, "phoneme_cache"),
    compute_input_seq_cache=True,
    print_step=25,
    use_language_weighted_sampler=False,
    print_eval=False,
    mixed_precision=False,
    output_path=output_path,
    datasets=dataset_config,
    characters=CharactersConfig(
        characters_class="TTS.tts.models.vits.VitsCharacters",
        pad="_",
        eos="&",
        bos="*",
        characters="abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZァアィイゥウェエォオカガキギクグケゲコゴサザシジスズセゼソゾタダチヂッツヅテデトドナニヌネノハバパヒビピフブプヘベペホボポマミムメモャヤュユョヨラリルレロワヲンヴ・！，．？−、。「」『』　 ー",
        punctuations="・！，．？−、。「」『』　 ",
        phonemes=None,
        is_unique=True,
    ),
)
ap = AudioProcessor(**config.audio.to_dict())

tokenizer, tokenizer_config = TTSTokenizer.init_from_config(config)

train_samples, eval_samples = load_tts_samples(dataset_config, eval_split=True)
print(len(train_samples))
print(len(eval_samples))

speaker_manager = SpeakerManager()
speaker_manager.set_ids_from_data(train_samples + eval_samples, parse_key="speaker_name")
config.model_args.num_speakers = speaker_manager.num_speakers

model = Vits(config=config, ap=ap, speaker_manager=speaker_manager, tokenizer=tokenizer)

print("training start")
wandb.init(project="yourTTS-jvs", sync_tensorboard=True)

# init the trainer and 🚀

trainer = Trainer(
    TrainerArgs(),
    config,
    output_path,
    model=model,
    train_samples=train_samples,
    eval_samples=eval_samples,
)

trainer.fit()

"""

dataset_path = "../jvs_ver1"
dataset_config = [BaseDatasetConfig(formatter="jvs", meta_file_train=None, path=dataset_path, language="ja")]

config = load_config(config_path)
# confirm json formmat
config.from_dict(config.to_dict())

# config["characters"]["is_unique"] = False
config["datasets"] = dataset_config
config.model_args["d_vector_file"] = speaker_path
# init model
model = setup_model(config)  # Vits(config, ap, tokenizer, speaker_manager, language_manager)


train_samples, eval_samples = load_tts_samples(
    config["datasets"],
    eval_split=True,
    eval_split_max_size=config.eval_split_max_size,
    eval_split_size=config.eval_split_size,
)

# # load parameters
# cp = torch.load(model_path, map_location=torch.device("cuda"))
# model_weights = cp["model"].copy()

# # TODO strict Falseでいいのか？
# model.load_state_dict(model_weights, strict=False)
# model = model.cuda()

# exit(1)

print("training start")
# init the trainer and 🚀
trainer = Trainer(
    TrainerArgs(), config, output_path, model=model, train_samples=train_samples, eval_samples=eval_samples
)
trainer.fit()

"""
