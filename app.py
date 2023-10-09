import pandas as pd
import numpy as np
import matplotlib
# matplotlib.use('macosx')
import gradio as gr
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from apscheduler.schedulers.background import BackgroundScheduler

ENG_COLS = [
    ("Model", "str"),
    ("FPB-acc", "number"),
    ("FPB-F1", "number"),
    ("FPB-missing", "number"),
    ("FiQA-SA-F1", "number"),
    ("FiQA-SA-missing", "number"),
    ("Headline-AvgF1", "number"),
    ("NER-EntityF1", "number"),
    ("ConvFinQA-EmAcc", "number"),
    ("FinQA-EmAcc", "number"),
    ("BigData22-Acc", "number"),
    ("BigData22-MCC", "number"),
    ("BigData22-missing", "number"),
    ("ACL18-Acc", "number"),
    ("ACL18-MCC", "number"),
    ("ACL18-missing", "number"),
    ("CIKM18-Acc", "number"),
    ("CIKM18-MCC", "number"),
    ("CIKM18-missing", "number"),
    ("FOMC-acc", "number"),
    ("FOMC-F1", "number"),
    ("FOMC-missing", "number"),
    ("FinerOrd-EntityF1", "number"),
    ("FinerOrd-F1", "number"),
    ("German-Acc", "number"),
    ("German-MCC", "number"),
    ("German-missing", "number"),
    ("Australian-Acc", "number"),
    ("Australian-MCC", "number"),
    ("Australian-missing", "number"),
    ("TSA-RMSE", "number"),
    ("TSA-missing", "number"),
    ("MLESG-F1", "number"),
    ("MLESG-missing", "number"),
    ("FSRL-entity-F1", "number"),
    ("FSRL-F1", "number"),
    ("CFA-acc", "number"),
    ("CFA-F1", "number"),
    ("CFA-missing", "number"),
    ("Finarg-ECCAUC-F1", "number"),
    ("Finarg-ECCAUC-missing", "number"),
    ("Finarg-ECCARC-F1", "number"),
    ("Finarg_ECCARC-missing", "number"),
    ("CD-Entity-F1", "number"),
    ("CD-F1", "number"),
    ("MultiFin-EN-acc", "number"),
    ("MultiFin-EN-F1", "number"),
    ("MultiFin-EN-missing", "number"),
    ("MA-acc", "number"),
    ("MA-F1", "number"),
    ("MA-missing", "number"),
    ("Causal20-sc-acc", "number"),
    ("Causal20-sc-F1", "number"),
    ("Causal20-sc-missing", "number"),
    ("TATQA-EmAcc", "number"),
    ("FNXL-entity-F1", "number"),
    ("FNXL-F1", "number"),
    ("FinRED-precision", "number"),
    ("FinRED-recall", "number"),
    ("FinRED-F1", "number"),
    ("ECTSUM-Rouge1", "number"),
    ("ECTSUM-Rouge2", "number"),
    ("ECTSUM-RougeL", "number"),
    ("ECTSUM-BertScore", "number"),
    ("ECTSUM-BARTScore", "number"),
    ("EDTSUM-Rouge1", "number"),
    ("EDTSUM-Rouge2", "number"),
    ("EDTSUM-RougeL", "number"),
    ("EDTSUM-BertScore", "number"),
    ("EDTSUM-BARTScore", "number"),
]

SPA_COLS = [
    ("Model", "str"),
    ("MultiFin-F1", "number"),
    ("MultiFin-Acc", "number"),
    ("FNS-Rouge1", "number"),
    ("FNS-Rouge2", "number"),
    ("FNS-RougeL", "number"),
    ("EFP-F1", "number"),
    ("EFP-Acc", "number"),
    ("EFPA-F1", "number"),
    ("EFPA-Acc", "number"),
    ("TSA-F1", "number"),
    ("TSA-Acc", "number"),
    ("FinanceES-F1", "number"),
    ("FinanceES-Acc", "number"),
]

CHI_COLS = [
    ("Model", "str"),
    ("AFQMC-Acc", "number"),
    ("AFQMC-F1", "number"),
    ("corpus-Acc", "number"),
    ("corpus-F1", "number"),
    ("stockA-Acc", "number"),
    ("stockA-F1", "number"),
    ("Fineval-Acc", "number"),
    ("Fineval-F1", "number"),
    ("NL-Acc", "number"),
    ("NL-F1", "number"),
    ("NL2-Acc", "number"),
    ("NL2-F1", "number"),
    ("NSP-Acc", "number"),
    ("NSP-F1", "number"),
    ("RE-Acc", "number"),
    ("RE-F1", "number"),
    ("FE-Acc", "number"),
    ("FE-F1", "number"),
    ("stockB-Acc", "number"),
    ("stockB-F1", "number"),
]


# Extract column names
eng_cols = [col_name for col_name, _ in ENG_COLS]
eng_cates = {
    "Sentiment Analysis": ["Model", "FPB-acc", "FPB-F1", "FPB-missing",
        "FiQA-SA-F1", "FiQA-SA-missing", "Headline-AvgF1", "TSA-RMSE",
        "TSA-missing", "FOMC-acc", "FOMC-F1", "FOMC-missing"],
    "NER": ["Model", "NER-EntityF1", "FinerOrd-EntityF1", "FinerOrd-F1"],
    "Number Understanding": ["Model", "FinQA-EmAcc", "ConvFinQA-EmAcc"],
    "Text Summarization": ["Model", "ECTSUM-Rouge1", "ECTSUM-Rouge2",
        "ECTSUM-RougeL", "ECTSUM-BertScore", "ECTSUM-BARTScore",
        "EDTSUM-Rouge1", "EDTSUM-Rouge2", "EDTSUM-RougeL", "EDTSUM-BertScore", "EDTSUM-BARTScore",],
    "Stock Movement Prediction": ["Model", "BigData22-Acc",
        "BigData22-MCC", "BigData22-missing", "ACL18-Acc", "ACL18-MCC",
        "ACL18-missing", "CIKM18-Acc", "CIKM18-MCC", "CIKM18-missing", ],
    "Credit Scoring": ["Model", "German-Acc", "German-MCC",
        "German-missing", "Australian-Acc", "Australian-MCC", "Australian-missing"],
}

spa_cols = [col_name for col_name, _ in SPA_COLS]
spa_cates = {
    "Sentiment Analysis": ["Model", "TSA-Acc", "TSA-F1", "FinanceES-Acc", "FinanceES-F1"],
    "Examination": ["Model", "EFP-Acc", "EFP-F1", "EFPA-Acc", "EFPA-F1"],
    "Classification": ["Model", "MultiFin-Acc", "MultiFin-F1"],
    "Text Summarization": ["Model", "FNS-Rouge1", "FNS-Rouge2", "FNS-RougeL",],
}

chi_cols = [col_name for col_name, _ in CHI_COLS]
chi_cates = {
    "Semantic matching": ["Model", "AFQMC-Acc", "AFQMC-F1", "corpus-Acc", "corpus-F1"],
    "Classification": ["Model", "NL-Acc", "NL-F1","NL2-Acc", "NL2-F1","NSP-Acc", "NSP-F1"],
    "Stock Movement Prediction": ["Model", "stockA-Acc", "stockA-F1"]
    "Examination": ["Model", "Fineval-Acc", "Fineval-F1"],
    "Relation Extraction": ["Model", "RE-Acc", "RE-F1"],
    "Sentiment Analysis": ["Model", "FE-Acc", "FE-F1", "stockB-Acc", "stockB-F1"],
}

def create_df_dict(lang, lang_cols, cates):
    # Load leaderboard data with column names
    leaderboard_df = pd.read_csv(f'{lang}_result.csv', names=lang_cols)
    leaderboard_df = leaderboard_df.sort_index(axis=1)
    # Move 'key' column to the front
    leaderboard_df = leaderboard_df[ ['Model'] + [ col for col in leaderboard_df.columns if col != 'Model' ] ]
    cols = leaderboard_df.columns
    types = ["str"] + ["number"] * (len(lang_cols)-1)

    # Split merged_df into subtask dataframes
    df_dict = {}
    for key, selected_columns in cates.items():
        df_dict[key] = leaderboard_df[selected_columns]
    return df_dict

df_lang = {
    "English": create_df_dict("english", eng_cols, eng_cates),
    "Spanish": create_df_dict("spanish", spa_cols, spa_cates),
    "Chinese": create_df_dict("chinese", chi_cols, chi_cates),
}


# Constants
TITLE = '<h1 align="center" id="space-title">🐲 PIXIU FLARE Leaderboard</h1>'
# TITLE = "Financial Natural Language Understanding and Prediction Evaluation Benchmark (FLARE) Leaderboard"
INTRODUCTION_TEXT = """📊 The PIXIU FLARE Leaderboard is designed to rigorously track, rank, and evaluate state-of-the-art models in financial Natural Language Understanding and Prediction. 

📈 Unique to FLARE, our leaderboard not only covers standard NLP tasks but also incorporates financial prediction tasks such as stock movement and credit scoring, offering a more comprehensive evaluation for real-world financial applications.

📚 Our evaluation metrics include, but are not limited to, Accuracy, F1 Score, ROUGE score, BERTScore, and Matthews correlation coefficient (MCC), providing a multidimensional assessment of model performance.

🔗 For more details, refer to our GitHub page [here](https://github.com/ChanceFocus/PIXIU).
"""


def create_data_interface(df):
    headers = df.columns
    types = ["str"] + ["number"] * (len(headers) - 1)

    return gr.components.Dataframe(
        value=df.values.tolist(),
        headers=[col_name for col_name in headers],
        datatype=types,
        max_rows=10,
    )

def plot_radar_chart(df, attributes, category_name):
    fig = go.Figure()

    for index, row in df.iterrows():
        model = row['Model']
        values = row[attributes].tolist()
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=attributes,
            fill='toself',
            name=model
        ))

    fig.update_layout(
        title="FLARE",
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 0.9]
            )),
        showlegend=True
    )

    return fig

def create_data_interface_for_aggregated(df, category_name):
    attributes = df.columns[1:]
    print (attributes)
    plt = plot_radar_chart(df, attributes, category_name)
    return plt

def create_lang_leaderboard(df_dict):
    new_df = pd.DataFrame()
    for key, df in df_dict.items():
        new_df["Model"] = df["Model"]
        tdf = df.replace('', 0)
        tdf = tdf[[val for val in tdf.columns if "Model" not in val]]
        if key == "Sentiment Analysis":
            tdf = tdf[[val for val in tdf.columns if "F1" in val]]
        elif key == "Classification":
            tdf = tdf[[val for val in tdf.columns if "F1" in val]]
        elif key == "Examination":
            tdf = tdf[[val for val in tdf.columns if "F1" in val]]
        elif key == "Stock Movement Prediction":
            tdf = tdf[[val for val in tdf.columns if "Acc" in val]]
        elif key == "Credit Scoring":
            tdf = tdf[[val for val in tdf.columns if "Acc" in val]]
        elif key == "Text Summarization":
            tdf = tdf[[val for val in tdf.columns if "Bert" in val or "Rouge" in val]]
        elif key == "Semantic matching":
            tdf = tdf[[val for val in tdf.columns if "Acc" in val]]
        elif key == "Relation Extraction":
            tdf = tdf[[val for val in tdf.columns if "Acc" in val]]
        print ("tdf")
        print (tdf)
        new_df[key] = tdf.values.mean(axis=1)
    print (new_df.values)

    plot = create_data_interface_for_aggregated(new_df, key)
    gr.Plot(plot)

    for key, df in df_dict.items():
        with gr.Tab(key):
            create_data_interface(df)

def launch_gradio():
    demo = gr.Blocks()

    with demo:
        gr.HTML(TITLE)
        gr.Markdown(INTRODUCTION_TEXT, elem_classes="markdown-text")
        for key, df_dict in df_lang.items():
            with gr.Tab(key):
                create_lang_leaderboard(df_dict)
        
    demo.launch()

scheduler = BackgroundScheduler()
scheduler.add_job(launch_gradio, "interval", seconds=3600)
scheduler.start()

# Launch immediately
launch_gradio()
