# Finetuning手順

## 1. JVSコーパスのダウンロード

```bash
cd ../
bash download_jvs.sh
```

## 2. 学習済みモデルのダウンロード

モデルをダウンロードします。`download/` ができて、その中にモデルが格納されます。
```bash
bash download_model.sh
```

## 3. d_vector_fileの作成（コーパスの音声特徴量の抽出）

exp/に格納されます。
```bash
bash extract_speaker_feature.sh
```

## 4. ModelのFinetuning

config.json の中の以下を書き換える必要がある

- datasets
- output_path
- d_vector_file

それぞれ、finetuning.pyの中で変更済み
