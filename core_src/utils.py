import os
from sqlalchemy import text
import pandas as pd
import json
import time
import datetime
import random
import string
import re
import base64


def write_log_file(text):
    file = open("../prompts_log.txt", "a")
    file.write(text)
    file.close()


def is_figure(file_name: str):
    if file_name[:3] == "fig" and file_name[-3:] == "png":
        return True
    return False


def is_csv(file_name: str):
    return True if file_name[-4:] == ".csv" else False


def is_sql(text: str):
    start_with = "sql_query:"
    return True if text[: len(start_with)] == start_with else False


def generate_session_id(length=8):
    # Get the current timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")

    # Generate a random string
    random_string = "".join(
        random.choices(string.ascii_letters + string.digits, k=length)
    )

    # Combine timestamp and random string
    session_id = f"{timestamp}{random_string}"
    return session_id

# Streamed response emulator
def response_generator(text=""):
    if text == "":

        text = random.choice(
            [
                "Hello there! How can I assist you today?",
                "Hi! Is there anything I can help you with?",
                "Do you need help?",
            ]
        )
    # collect_responses += [text]
    for word in text.split():
        yield word + " "
        time.sleep(0.05)


def check_empty_plot(figure):
    def check_plot(plot):
        if "x" in plot and plot["x"] is None:
            return True
        elif "y" in plot and plot["y"] is None:
            return True
        elif "x" in plot and len(plot["x"]) == 0:
            return True
        elif "y" in plot and len(plot["y"]) == 0:
            return True
        # for plots like sunburst and treemap
        elif "values" in plot and plot["values"] is None:
            return True
        elif "values" in plot and len(plot["values"]) == 0:
            return True
        # for plots like line_mapbox
        elif "lat" in plot and len(plot["lat"]) == 0:
            return True
        return False
    # check if it is an empty go.Figure
    if figure.data == tuple():
        return True
    return all(check_plot(plot) for plot in figure.data)

