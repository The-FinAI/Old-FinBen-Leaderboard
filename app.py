from apscheduler.schedulers.background import BackgroundScheduler
import gradio as gr
import pandas as pd

# Load leaderboard data
leaderboard_df = pd.read_csv('leaderboard.csv').transpose()

# Constants
TITLE = "Leaderboard"
INTRODUCTION_TEXT = "This is the leaderboard."

def launch_gradio():
    demo = gr.Blocks()

    with demo:
        gr.HTML(TITLE)
        gr.Markdown(INTRODUCTION_TEXT, elem_classes="markdown-text")

        # Create a gradio table from pandas dataframe
        leaderboard_table = gr.components.Dataframe(
            value=leaderboard_df,
            max_rows=5,
            elem_id="leaderboard-table",
        )

    demo.launch()

scheduler = BackgroundScheduler()
scheduler.add_job(launch_gradio, "interval", seconds=3600)
scheduler.start()

# Launch immediately
launch_gradio()
