# Finetuning手順

## 1. JVSコーパスのダウンロード

```bash
cd ../
bash download_jvs.sh
```

## 2. Speaker Embedding model やその他データののダウンロード

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

## 3. d_vector_fileの作成（コーパスの音声特徴量の抽出）

> もし、前のセクションでspeaker.jsonをダウンロードできているなら、飛ばしても大丈夫です。
> ちなみに、JVSのみのspkear.jsonもダウンロードしているので、適時使ってください。


exp/に格納されます。
```bash
bash extract_speaker_feature.sh
```

## 4. WandBへのログイン
WandBを使ってログを見れるようにしています。

以下のコマンドを使ってログインして見れるようにしてください

```bash
wandb login
```

## 5. ModelのTrianing

`training_yourTTS.py`でパスを修正して実行してください。

```bash
python training_yourTTS.py
```
