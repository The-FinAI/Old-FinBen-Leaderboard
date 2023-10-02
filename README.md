---
title: FLARE
emoji: 🐠
colorFrom: pink
colorTo: pink
sdk: gradio
sdk_version: 3.34.0
app_file: app.py
pinned: false
license: mit
---

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference

## Add New Language

1. Add new `[LAN]_result.csv`, which should be format like

|Model|Task Metric 1|Task Metric 2|
|---|---|---|
|GPT-4|0.28|0|

2. Add new COL variable on `app.py`

```python
SPA_COLS = [
    ("Model", "str"),
    ("MultiFin-F1", "number"),
]
```

3. Add new Col categorization on `app.py`

```python
spa_cols = [col_name for col_name, _ in SPA_COLS]
spa_cates = {
    "Sentiment Analysis": ["Model", "TSA-Acc", "TSA-F1", "FinanceES-Acc", "FinanceES-F1"],
    "Examination": ["Model", "EFP-Acc", "EFP-F1", "EFPA-Acc", "EFPA-F1"],
    "Classification": ["Model", "MultiFin-Acc", "MultiFin-F1"],
    "Text Summarization": ["Model", "FNS-Rouge1", "FNS-Rouge2", "FNS-RougeL",],
}
```

4. Add new key to lan dict on `app.py` 

```python
df_lang = {
    "English": create_df_dict("english", eng_cols, eng_cates),
    "Spanish": create_df_dict("spanish", spa_cols, spa_cates),
}
```

5. If new categories need to define new column selection rules, add it like:

```python
elif key == "Credit Scoring":
    tdf = tdf[[val for val in tdf.columns if "Acc" in val]]
elif key == "Text Summarization":
    tdf = tdf[[val for val in tdf.columns if "Bert" in val or "Rouge" in val]]
```
