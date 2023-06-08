from apscheduler.schedulers.background import BackgroundScheduler
import gradio as gr
import pandas as pd

def show_leaderboard():
    # Load the leaderboard data from the CSV
    df = pd.read_csv("leaderboard.csv")

    # Convert the dataframe to a markdown table for easy display in Gradio
    markdown_table = df.to_markdown()

    return markdown_table

def launch_gradio():
    # Define Gradio interface
    iface = gr.Interface(
        fn=show_leaderboard, 
        inputs=[], 
        outputs="markdown"
    )

    # Run the interface
    iface.queue(concurrency_count=40).launch()

# Define the scheduler
scheduler = BackgroundScheduler()
scheduler.add_job(launch_gradio, "interval", seconds=3600)
scheduler.start()

# Call launch_gradio to start the first Gradio app immediately
launch_gradio()
