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

output_path = "exp_yourTTS_vctk_jvs"
d_vector_path = "exp/spekaers.json"
dataset_config = [
    BaseDatasetConfig(
        formatter="jvs",
        dataset_name="jvs",
        meta_file_train="train",
        meta_file_val="dev",
        path="../jvs_ver1",
        language="ja",
        ignored_speakers=["jvs100", "jvs096"],
    ),
    BaseDatasetConfig(
        formatter="vctk",
        meta_file_train="train",
        meta_file_val="dev",
        language="en-us",
        ignored_speakers=None,
        path="../../vctk/VCTK/",
    ),
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
    trim_db=45,
    mel_fmin=0,
    mel_fmax=None,
    spec_gain=1.0,
    signal_norm=True,
    do_amp_to_db_linear=False,
    resample=True,
)


vitsArgs = VitsArgs(
    use_language_embedding=False,
    use_speaker_embedding=False,
    use_d_vector_file=True,
    d_vector_file=d_vector_path,
    d_vector_dim=512,
    use_sdp=True,
    num_layers_text_encoder=10,
    inference_noise_scale_dp=0.8,
    speakers_file=None,
    speaker_embedding_channels=512,
)

config = VitsConfig(
    model_args=vitsArgs,
    audio=audio_config,
    run_name="yourTTS_jvs_vctk",
    run_description="🐸Coqui trainer run.",
    batch_size=16,
    eval_batch_size=16,
    batch_group_size=0,
    num_loader_workers=8,
    num_eval_loader_workers=8,
    run_eval=True,
    test_delay_epochs=5,
    epochs=1000,
    text_cleaner="multilingual_cleaners",
    use_phonemes=False,
    phoneme_language="jp-ja",
    phoneme_cache_path=os.path.join(output_path, "phoneme_cache"),
    compute_input_seq_cache=True,
    print_step=25,
    use_language_weighted_sampler=False,
    print_eval=True,
    mixed_precision=False,
    output_path=output_path,
    datasets=dataset_config,
    grad_clip=[5.0, 5.0],
    min_audio_len=32 * 256 * 4,
    max_audio_len=160000,
    characters=CharactersConfig(
        characters_class="TTS.tts.models.vits.VitsCharacters",
        pad="_",
        eos="&",
        bos="*",
        characters="abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZァアィイゥウェエォオカガキギクグケゲコゴサザシジスズセゼソゾタダチヂッツヅテデトドナニヌネノハバパヒビピフブプヘベペホボポマミムメモャヤュユョヨラリルレロワヲンヴ・！，．？ー、。「」『』　!'(),-.:;? ",
        punctuations="・！，．？ー、。「」『』　!'(),-.:;? ",
        phonemes=None,
        is_unique=True,
    ),
    test_sentences=[
        ["マタ、トージノヨーニ、ゴダイミョーオートヨバレル、シュヨーナミョーオーノチューオーニハイサレルコトモオーイ。"],
        ["ニューイングランドフーワ、ギューニューヲベーストシタ、シロイクリームスープデアリ、ボストンクラムチャウダートモヨバレル。"],
        ["コンピュータゲームノメーカーヤ、ギョーカイダンタイナドニカンレンスルジンブツノカテゴリ。"],
        ["サービスマネージャードーニューエキノタメ、オーイマチエキカラ、エンカクカンリシテイル。"],
        ["シルバーサーファーシューゲキジケンマデニ、リチャーズワ、チームメートトモニ、コクサイテキニスーパーヒーロー、オヨビ、ユーメージントシテ、ニンチサレテイル。"],
        ["ホッカイドーニイッタラ、ヤッパリ、ウミノコーヲタベナイト、イッタイミガナイデショー。"],
        ["メサキノリエキダケニトラワレテワイケナイ。"],
        ["イザカヤデ、ヘンナオッサンニカラマレタ。"],
        ["テニスニモアルケド、ヨンダイタイカイッテナニ。"],
        ["トーストニマイト、オレンジジュースヲオネガイシマス。"],
    ],
)
ap = AudioProcessor(**config.audio.to_dict())

tokenizer, tokenizer_config = TTSTokenizer.init_from_config(config)

train_samples, eval_samples = load_tts_samples(
    dataset_config,
    eval_split=True,
)
print(len(train_samples))
print(len(eval_samples))

speaker_manager = SpeakerManager(d_vectors_file_path=d_vector_path)
speaker_manager.set_ids_from_data(train_samples + eval_samples, parse_key="speaker_name")
config.model_args.num_speakers = speaker_manager.num_speakers
print(f"speaker number:{speaker_manager.num_speakers}")

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
