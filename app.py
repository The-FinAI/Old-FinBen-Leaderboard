from apscheduler.schedulers.background import BackgroundScheduler
import gradio as gr
import pandas as pd

# Load leaderboard data
leaderboard_df = pd.read_csv('leaderboard.csv')

# Constants
TITLE = "Leaderboard"
INTRODUCTION_TEXT = "This is the leaderboard."

def launch_gradio():
    demo = gr.Blocks()

    with demo:
        gr.HTML(TITLE)
        gr.Markdown(INTRODUCTION_TEXT, elem_classes="markdown-text")

        leaderboard_table = gr.Interface.load("gradio.dataframe", leaderboard_df)

    demo.launch()

scheduler = BackgroundScheduler()
scheduler.add_job(launch_gradio, "interval", seconds=3600)
scheduler.start()

# Launch immediately
launch_gradio()
