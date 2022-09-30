# Finetuning手順

## 1. JVSコーパスのダウンロード

```bash
cd /path/to/TTS/recipes/jvs
bash download_jvs.sh
```

## 2. VCTK コーパスのダウンロード

```bash
cd /path/to/TTS/recipes/vctk
bash download_vctk.sh
```


## ３. Speaker Embedding model やその他データののダウンロード

以下のファイルをダウンロードします。`download/` ができて、その中にモデルが格納されます。
- dataset_config.json
  dataset用のconfig.json
  extract_speaker_feature.shを叩くときに必要
  
- jvs_vctk_speakers.json
  JVSとVCTKの混合データセットのEmbedされた特徴量
  予め計算してjson として保存している
　trainingの際に必要

- config_se.json 
  Speaker Embedding model config
　trainingの際に必要

- SE_checkpoint.pth.tar 
  Speaker Embedding model
  trianingの際に必要
```bash
bash download_model.sh
```
## 4. d_vector_fileの作成（コーパスの音声特徴量の抽出）

> もし、前のセクションでspeaker.jsonをダウンロードできているなら、飛ばしても大丈夫です。
> ちなみに、JVSのみのspkear.jsonもダウンロードしているので、適時使ってください。

exp/に格納されます。
```bash
bash extract_speaker_feature.sh
```

## 5. WandBへのログイン
WandBを使ってログを見れるようにしています。

以下のコマンドを使ってログインして見れるようにしてください
```bash
wandb login
```
## 6. Model Training
 
パラメータなどは `yourTTS_train.py`に記述してあります。
パラメータ調整は、論文内で一番良かったものにできるだけ合わせています。

`training_yourTTS.py`でパスを修正して実行してください。

```bash
    python yourTTS_train.py
```

## inference 

以下のコマンドで推論できます。model_pathとconfig_pathは適時変更してください

```bash

python inference.py \
    --model_path exp_yourTTS_vctk_jvs/yourTTS_finetuning-September-29-2022_10+34AM-e0a82820/best_model.pth \
    --config_path exp_yourTTS_vctk_jvs/yourTTS_finetuning-September-29-2022_10+34AM-e0a82820/config.json \
    --SE_config_path download/config_se.json \
    --SE_model_path download/SE_checkpoint.pth.tar \
    --target_path sample_wav/asano.wav \
    --source_path sample_wav/fukushima.wav \
    --tts_speaker exp/jvs_vctk_speakers.json \
    --output_path ./ \
    --language ja
```