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
    ("EDTSUM-rouge1", "number"),
    ("EDTSUM-rouge2", "number"),
    ("EDTSUM-rougeL", "number"),
]
TYPES = [col_type for _, col_type in COLS]
TYPES_AUTO = [col_type for _, col_type in COLS_AUTO]

# Extract column names
cols = [col_name for col_name, _ in COLS]
cols_auto = [col_name for col_name, _ in COLS_AUTO]

# Load leaderboard data with column names
leaderboard_df = pd.read_csv('leaderboard.csv', names=cols)
leaderboard_auto_df = pd.read_csv('leaderboard_auto.csv', names=cols_auto)
common_cols = list(set(cols) & set(cols_auto))

# Merge dataframes and replace NaN values with an empty string
merged_df = pd.merge(
    leaderboard_df, leaderboard_auto_df, how="outer", on=common_cols).fillna("")

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
df_number_understanding = merged_df[["Model", "FinQA-EmAcc"]]


df_dict = {
    "Sentiment Analysis": df_sentiment_analysis,
    "Stock Movement Prediction": df_stock_movement_prediction,
    "NER": df_ner,
    "Credit Scoring": df_credit_scoring,
    "Number Understanding": df_number_understanding,
}


# Constants
TITLE = "Financial Natural Language Understanding and Prediction Evaluation Benchmark (FLARE) Leaderboard"
INTRODUCTION_TEXT = "The leaderboard shows the performance of various models in financial natural language understanding and prediction tasks."

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
    N = len(attributes)

    # Compute the angle of each axis in the plot
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]

    # Create a radar/spider chart
    plt.figure(figsize=(10, 7))
    ax = plt.subplot(111, polar=True)
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    
    # Set labels
    plt.xticks(angles[:-1], attributes, color='grey', size=10)

    # Plot line for each model
    for _, row in df.iterrows():
        model_name = row['Model']
        values = row.drop('Model').values.tolist()
        values += values[:1]
        ax.plot(angles, values, linewidth=2, linestyle='solid', label=model_name)
        ax.fill(angles, values, alpha=0.1)

    ax.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    plt.title(category_name, size=20, color='black', y=1.1)
    plt.close()

    return plt

def create_data_interface_for_aggregated(df, category_name):
    attributes = df.columns[1:]
    plt = plot_radar_chart(df, attributes, category_name)
    return plt

def launch_gradio():
    demo = gr.Blocks()

    with demo:
        gr.HTML(TITLE)
        gr.Markdown(INTRODUCTION_TEXT, elem_classes="markdown-text")

        for key, df in df_dict.items():
            with gr.Tab(key):
                create_data_interface(df)
                plot = create_data_interface_for_aggregated(df, key)
                gr.Plot(plot)

    demo.launch()

scheduler = BackgroundScheduler()
scheduler.add_job(launch_gradio, "interval", seconds=3600)
scheduler.start()

# Launch immediately
launch_gradio()
