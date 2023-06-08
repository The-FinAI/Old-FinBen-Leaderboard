from apscheduler.schedulers.background import BackgroundScheduler
import gradio as gr
import pandas as pd

# Load leaderboard data
leaderboard_df = pd.read_csv('leaderboard.csv')

# Constants
TITLE = "Financial Natural Language Understanding and Prediction Evaluation Benchmark (FLARE) Leaderboard"
INTRODUCTION_TEXT = "The leaderboard shows the performance of various models in financial natural language understanding and prediction tasks."

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

TYPES = [col_type for _, col_type in COLS]

def launch_gradio():
    demo = gr.Blocks()

    with demo:
        gr.HTML(TITLE)
        gr.Markdown(INTRODUCTION_TEXT, elem_classes="markdown-text")

        # Create a gradio table from pandas dataframe
        leaderboard_table = gr.components.Dataframe(
            value=leaderboard_df.values.tolist(),
            headers=[col_name for col_name, _ in COLS],
            datatype=TYPES,
            max_rows=5,
            elem_id="leaderboard-table",
        )

    demo.launch()

scheduler = BackgroundScheduler()
scheduler.add_job(launch_gradio, "interval", seconds=3600)
scheduler.start()

# Launch immediately
launch_gradio()
