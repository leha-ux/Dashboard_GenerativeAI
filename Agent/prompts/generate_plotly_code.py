'''Today is {today_date}.
You are provided with a pandas dataframe (df) with {num_rows} rows and {num_columns} columns.
This is the metadata of the dataframe:
{df_head}.

Here is the pandas code that was created earlier to answer the following question: {question}. The Code returns a pandas dataframe.
Code: 
{pandas_code}
It is now your task to visualize the answer for the question ({question}) using the python library plotly. 
Use: import plotly.graph_objects as go
The plotly figure will be shown in a dashboard.
Do not return a dataframe. Return a plotly figure to visualize the dataframe. The plotly figure should parsed to a html string 
Use: import plotly.io as pio
The maximum height of the figure is height=425px
Return the python code and make sure to prefix the requested python code with {START_CODE_TAG} exactly and suffix the code with {END_CODE_TAG} exactly to visualize the answer to the following question:
'''



from datetime import date

from Agent.constants import END_CODE_TAG, START_CODE_TAG

from .base import Prompt


class GeneratePlotlyCodePrompt(Prompt):
    """Prompt to generate Python code (Plotly)"""

    text: str = """
Today is {today_date}.
You are provided with a pandas dataframe (df) with {num_rows} rows and {num_columns} columns.
This is the metadata of the dataframe:
{df_head}.

Here is the pandas code that was created earlier to answer the following question: {question}. The Code returns a pandas dataframe.
Code: 
{pandas_code}
It is now your task to visualize the answer for the question ({question}) using the python library plotly. 
Use: import plotly.graph_objects as go
The plotly figure will be shown in a dashboard.
Do not return a dataframe. Return a plotly figure to visualize the dataframe. The plotly figure should parsed to a html string 
Use: import plotly.io as pio
The maximum height of the figure is height=425px
Return the python code and make sure to prefix the requested python code with {START_CODE_TAG} exactly and suffix the code with {END_CODE_TAG} exactly to visualize the answer to the following question:
""" 

    def __init__(self, **kwargs):
        super().__init__(
            **kwargs,
            START_CODE_TAG=START_CODE_TAG,
            END_CODE_TAG=END_CODE_TAG,
            today_date=date.today()
        )
