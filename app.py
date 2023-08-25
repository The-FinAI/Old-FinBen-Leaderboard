import pandas as pd
import numpy as np
import matplotlib
# matplotlib.use('macosx')
import gradio as gr
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from apscheduler.schedulers.background import BackgroundScheduler

COLS = [
    ("Model", "str"),
    ("FPB-acc", "number"),
    ("FPB-F1", "number"),
    ("FiQA-SA-F1", "number"),
    ("Headline-AvgF1", "number"),
    ("NER-EntityF1", "number"),
    ("FinQA-EmAcc", "number"),
    ("ConvFinQA-EmAcc", "number"),
    ("BigData22-Acc", "number"),
    ("BigData22-MCC", "number"),
    ("ACL18-Acc", "number"),
    ("ACL18-MCC", "number"),
    ("CIKM18-Acc", "number"),
    ("CIKM18-MCC", "number")
]

COLS_AUTO = [
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
    ("ECTSUM-rouge1", "number"),
    ("ECTSUM-rouge2", "number"),
    ("ECTSUM-rougeL", "number"),
    ("ECTSUM-BertScore", "number"),
    ("ECTSUM-BARTScore", "number"),
    ("EDTSUM-rouge1", "number"),
    ("EDTSUM-rouge2", "number"),
    ("EDTSUM-rougeL", "number"),
    ("EDTSUM-BertScore", "number"),
    ("EDTSUM-BARTScore", "number"),
]
TYPES = [col_type for _, col_type in COLS]
TYPES_AUTO = [col_type for _, col_type in COLS_AUTO]

# Extract column names
cols = [col_name for col_name, _ in COLS]
cols_auto = [col_name for col_name, _ in COLS_AUTO]

# Load leaderboard data with column names
leaderboard_df = pd.read_csv('leaderboard.csv', names=cols)
leaderboard_auto_df = pd.read_csv('present_result.csv', names=cols_auto)
common_cols = list(set(cols) & set(cols_auto))

# Merge dataframes and replace NaN values with an empty string
# merged_df = pd.merge(
#     leaderboard_df, leaderboard_auto_df, how="outer", on=common_cols).fillna("")
merged_df = leaderboard_auto_df

merged_df = merged_df.sort_index(axis=1)

# Move 'key' column to the front
merged_df = merged_df[ ['Model'] + [ col for col in merged_df.columns if col != 'Model' ] ]
merged_cols = merged_df.columns
merged_types = ["str"] + ["number"] * (len(merged_cols)-1)

# Split merged_df into subtask dataframes
df_sentiment_analysis = merged_df[["Model", "FPB-acc", "FPB-F1", "FPB-missing",
    "FiQA-SA-F1", "FiQA-SA-missing", "Headline-AvgF1", "TSA-RMSE",
    "TSA-missing", "FOMC-acc", "FOMC-F1", "FOMC-missing"]]
df_stock_movement_prediction = merged_df[["Model", "BigData22-Acc",
    "BigData22-MCC", "BigData22-missing", "ACL18-Acc", "ACL18-MCC",
    "ACL18-missing", "CIKM18-Acc", "CIKM18-MCC", "CIKM18-missing", ]]
df_ner = merged_df[["Model", "NER-EntityF1", "FinerOrd-EntityF1", "FinerOrd-F1"]]
df_credit_scoring = merged_df[["Model", "German-Acc", "German-MCC",
    "German-missing", "Australian-Acc", "Australian-MCC", "Australian-missing"]]
df_number_understanding = merged_df[["Model", "FinQA-EmAcc", "ConvFinQA-EmAcc"]]
df_text_summarization = merged_df[["Model", "ECTSUM-rouge1", "ECTSUM-rouge2",
    "ECTSUM-rougeL", "ECTSUM-BertScore", "ECTSUM-BARTScore",
    "EDTSUM-rouge1", "EDTSUM-rouge2", "EDTSUM-rougeL", "EDTSUM-BertScore", "EDTSUM-BARTScore",]]


df_dict = {
    "Sentiment Analysis": df_sentiment_analysis,
    "NER": df_ner,
    "Number Understanding": df_number_understanding,
    "Text Summarization": df_text_summarization,
    "Stock Movement Prediction": df_stock_movement_prediction,
    "Credit Scoring": df_credit_scoring,
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

def launch_gradio():
    demo = gr.Blocks()

    with demo:
        gr.HTML(TITLE)
        gr.Markdown(INTRODUCTION_TEXT, elem_classes="markdown-text")
        new_df = pd.DataFrame()
        for key, df in df_dict.items():
            new_df["Model"] = df["Model"]
            tdf = df.replace('', 0)
            if key == "Sentiment Analysis":
                tdf = tdf[[val for val in tdf.columns if "F1" in val]]
            elif key == "Stock Movement Prediction":
                tdf = tdf[[val for val in tdf.columns if "Acc" in val]]
            elif key == "Credit Scoring":
                tdf = tdf[[val for val in tdf.columns if "Acc" in val]]
            elif key == "Text Summarization":
                tdf = tdf[[val for val in tdf.columns if "Bert" in val]]
            print (tdf)
            new_df[key] = tdf.values[:, 1:].mean(axis=1)
        print (new_df)

        plot = create_data_interface_for_aggregated(new_df, key)
        gr.Plot(plot)

        for key, df in df_dict.items():
            with gr.Tab(key):
                create_data_interface(df)

    demo.launch()

scheduler = BackgroundScheduler()
scheduler.add_job(launch_gradio, "interval", seconds=3600)
scheduler.start()

# Launch immediately
launch_gradio()
