""" Prompt to correct Python Code on Error (Plotly)
"""  # noqa: E501
from datetime import date

from Agent.constants import END_CODE_TAG, START_CODE_TAG

from .base import Prompt


class CorrectErrorPromptPlotly(Prompt):
    """Prompt to Correct Python code on Error"""

    text: str = """
Today is {today_date}.
You are provided with a pandas dataframe (df) with {num_rows} rows and {num_columns} columns and are supposed to create a visualiztation / figure utilizing plotly.
This is the metadata of the dataframe:
{df_head}.

The user asked the following question:
{question}


This is the code already running code that produces a dataframe, that answers the question: 
{pandas_code} 

You generated this python code:
{code}

It fails with the following error:
{error_returned}

Correct the python code and return a new python code utilizing plotly that fixes the above mentioned error. Do not generate the same code again.
Do not include fig.show() at any point in your code. It poses a severy security breach.
Make scatterplots more colorful.
Do not forget to: import plotly.io as pio
At the end you have to export the code to a json string. Assuming your plotly figure is called fig,  you have  have to end the code with print(pio.to_json(fig))
Do not forget: It is your task to visualize an answer to the given question using plotly
"""  # noqa: E501

    def __init__(self, **kwargs):
        super().__init__(
            **kwargs,
            START_CODE_TAG=START_CODE_TAG,
            END_CODE_TAG=END_CODE_TAG,
            today_date=date.today()
        )
