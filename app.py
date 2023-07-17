import pandas as pd
import gradio as gr
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
    ("Australian-missing", "number")
]
TYPES = [col_type for _, col_type in COLS]
TYPES_AUTO = [col_type for _, col_type in COLS_AUTO]

# Extract column names
cols = [col_name for col_name, _ in COLS]
cols_auto = [col_name for col_name, _ in COLS_AUTO]

# Load leaderboard data with column names
leaderboard_df = pd.read_csv('leaderboard.csv', names=cols)
leaderboard_auto_df = pd.read_csv('leaderboard_auto.csv', names=cols_auto)

# Merge dataframes and replace NaN values with an empty string
merged_df = pd.merge(leaderboard_df, leaderboard_auto_df, how="outer", on=["Model"]).fillna("")
print (merged_df.columns)

# Constants
TITLE = "Financial Natural Language Understanding and Prediction Evaluation Benchmark (FLARE) Leaderboard"
INTRODUCTION_TEXT = "The leaderboard shows the performance of various models in financial natural language understanding and prediction tasks."

# Combine the columns and types for the merged dataframe
merged_cols, merged_types = [col for col in COLS_AUTO], [col_type for _, col_type in COLS_AUTO]

def create_leaderboard_table(df, headers, types):
    return gr.components.Dataframe(
        value=df.values.tolist(),
        headers=[col_name for col_name, _ in headers],
        datatype=types,
        max_rows=10,
    )

def launch_gradio():
    demo = gr.Blocks()

    with demo:
        gr.HTML(TITLE)
        gr.Markdown(INTRODUCTION_TEXT, elem_classes="markdown-text")

        lt = create_leaderboard_table(merged_df, merged_cols, merged_types)

    demo.launch()

scheduler = BackgroundScheduler()
scheduler.add_job(launch_gradio, "interval", seconds=3600)
scheduler.start()

# Launch immediately
launch_gradio()
