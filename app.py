import pandas as pd
import gradio as gr
from apscheduler.schedulers.background import BackgroundScheduler

# Extract column names
cols = [col_name for col_name, _ in COLS]
cols_auto = [col_name for col_name, _ in COLS_AUTO]

# Load leaderboard data with column names
leaderboard_df = pd.read_csv('leaderboard.csv', names=cols)
leaderboard_auto_df = pd.read_csv('leaderboard_auto.csv', names=cols_auto)

# Merge dataframes and replace NaN values with an empty string
merged_df = pd.merge(leaderboard_df, leaderboard_auto_df, how='outer', on=["Model"]).fillna("")

# Constants
TITLE = "Financial Natural Language Understanding and Prediction Evaluation Benchmark (FLARE) Leaderboard"
INTRODUCTION_TEXT = "The leaderboard shows the performance of various models in financial natural language understanding and prediction tasks."

# Combine the columns and types for the merged dataframe
merged_cols = COLS + [col for col in COLS_AUTO if col not in COLS]
merged_types = TYPES + [col_type for _, col_type in COLS_AUTO if (_, col_type) not in COLS]

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
