# Import necessary libraries and modules
import os
from textwrap import dedent
import ast
import dash
import dash_table
import dash_bootstrap_components as dbc
from dash.exceptions import PreventUpdate
from dash import dcc
from dash import html
from dash.dash import no_update
from dash.dependencies import Input, Output, State
import pandas as pd
import plotly.express as px
import openai
from Agent import Agent 
from Agent.llm.openai import OpenAI
import json
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
import time
from dash_extensions.enrich import Output, DashProxy, Input, MultiplexerTransform


# Prepare a placeholder string for explanation
explanation = """
*The code used for the visualization will be displayed here. Start by asking a question.*`
"""

# Import the dotenv module and load environment variables
from dotenv import load_dotenv
load_dotenv()

# Define a function to convert dictionary to a plotly figure and display
def parse_dict_to_plotly(json_string):
    # Load json data
    data_dict = json.loads(json_string)

    # Extract data from json
    data = data_dict['data'][0]
    layout = data_dict['layout']
    
    # Extract table values from data
    header_values = data['header']['values']
    cell_values = data['cells']['values']
    fill_color = data['cells']['fill']['color']
    header_fill_color = data['header']['fill']['color']

    # Create table trace
    trace = go.Table(
        header=dict(values=header_values, fill_color=header_fill_color, align='left'),
        cells=dict(values=cell_values, fill_color=fill_color, align='left'))

    # Create layout
    layout = go.Layout(layout)

    # Create figure with the trace and layout, then display it
    fig = go.Figure(data=[trace], layout=layout)
    fig.show()


# Define a function that appends an element to a list and returns the list
def append_element(lst, element):
    lst.append(element)
    return lst

# Define a function that creates and returns a header for the application using dash-bootstrap-components
def Header(name, app):
    title = html.H1(name, style={"margin-top": 5})
    return dbc.Row([dbc.Col(title, md=8)])

# Define a function that creates and returns a text box for a chat interface
def textbox(text, box="AI", name="Philippe"):
    # Remove specific names from the text
    text = text.replace(f"{name}:", "").replace("You:", "")
    # Define a base style for the text box
    style = {
        "max-width": "60%",
        "width": "max-content",
        "padding": "5px 10px",
        "border-radius": 25,
        "margin-bottom": 20,
    }

    if box == "user":
        # Modify style for user's text box and return it
        style["margin-left"] = "auto"
        style["margin-right"] = 0
        return dbc.Card(text, style=style, body=True, color="primary", inverse=True)

    elif box == "AI":
        # Modify style for AI's text box, add a thumbnail, and return it
        style["margin-left"] = 0
        style["margin-right"] = "auto"
        thumbnail = html.Img(
            src=app.get_asset_url("chatbot_image.png"),
            style={
                "border-radius": 50,
                "height": 36,
                "margin-right": 5,
                "float": "left",
            },
        )
        textbox = dbc.Card(text, style=style, body=True, color="light", inverse=False)
        return html.Div([thumbnail, textbox])

    else:
        # Raise an error if the input for 'box' is neither 'user' nor 'AI'
        raise ValueError("Incorrect option for `box`.")

# Specify the path to your Excel file
excel_file = os.path.join(os.path.dirname(__file__), "assets", "Financial_Sample.xlsx")

# Specify the name or index of the worksheet you want to import
worksheet_name = 'Sheet1'  

# Import the specified worksheet as a Pandas DataFrame
df = pd.read_excel(excel_file, sheet_name=worksheet_name)

# Authenticate with OpenAI using the key from environment variables
openai.api_key = os.getenv("OPENAI_KEY")

# Instantiate a language learning model with the OpenAI token
llm = OpenAI(api_token=os.getenv("OPENAI_KEY"))

# Instantiate an Agent with the language learning model
AgentDF = Agent(llm)


# Create a Dash application with Bootstrap theme
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.config.suppress_callback_exceptions=True  # Suppress exceptions for callback output
server = app.server  # Get the Flask server instance

# Define the conversation div for chat display, with CSS for scrolling and layout
conversation = html.Div(
    html.Div(id="display-conversation"),
    style={
        "overflow-y": "auto",
        "display": "flex",
        "height": "calc(90vh - 132px)",
        "flex-direction": "column-reverse",
    },
)

# Define the control input group for user chat input and submission
controls = dbc.InputGroup(
    children=[
        dbc.Input(id="user-input", placeholder="Write to the chatbot...", type="text"),
        dbc.InputGroupAddon(dbc.Button("Submit", id="submit"), addon_type="append"),
    ]
)

# Check if a string is a dictionary with nested dictionaries
def check_nested_dict(dict_str):
    try:
        dict_obj = json.loads(dict_str)  # Parse string to dictionary
        for value in dict_obj.values():
            if isinstance(value, dict):  # Check if any value is a dictionary
                return True
        return False
    except Exception as e: 
        print(e)
        return e

# Create a multi-column datatable from a string representing a dictionary
# Aggregate several columns by one key
def create_several_columns(dict_string):
    try: 
        dict_table = json.loads(dict_string)  # Parse string to dictionary
        table_data2 = []
        all_keys = set()

        # Extract all unique keys from the dictionary values
        for key, value in dict_table.items():
            all_keys.update(value.keys())

        # Generate columns dynamically based on all_keys
        columns = [{'name': '', 'id': 'Grouped by'}]
        for key in all_keys:
            columns.append({'name': key, 'id': key})

        # Generate table_data2 dynamically based on keys and values in dict_table
        for group in dict_table.keys():
            row = {"Grouped by": group}
            for key in all_keys:
                if key in dict_table[group]:
                    row[key] = dict_table[group][key]
                else:
                    row[key] = 0
            table_data2.append(row)

        # Create the DataTable with custom styles
        table2 = dash_table.DataTable(
        id='table2',
        columns=columns,
        data=table_data2,
        style_table={
            'width': '100%',
            'margin': 'auto',
            'maxWidth': '500px',
            'fontFamily': 'Arial, sans-serif',
            'border': '1px solid #ddd',
            'borderCollapse': 'collapse',
            'borderRadius': '5px',
            'overflowX': 'auto',
        },
        style_header={
            'backgroundColor': '#f2f2f2',
            'fontWeight': 'bold',
            'textTransform': 'uppercase'
        },
        style_cell={
            'padding': '10px',
            'textAlign': 'left',
            'whiteSpace': 'normal',
            'lineHeight': '1.5'
        },
        style_data={
            'backgroundColor': 'white',
            'color': '#333'
        },   
        )
        return html.Div([html.H4(''), table2], style={'padding-bottom': '20px'})  # Add a 5px gap below the div
    except Exception as e: 
        print(e)
        return None

# Define a function to create a two-column datatable from a string representing a dictionary
def create_single_column(data_str):
    try:
        data_dict = json.loads(data_str)  # Parse string to dictionary
        # Create list of dictionaries representing table data
        table_data = [{"Category": key, "Value": value} for key, value in data_dict.items()]
        # Create DataTable with custom styles
        table = dash_table.DataTable(
            id='table',
            columns=[{'name': 'Category', 'id': 'Category'}, {'name': 'Value', 'id': 'Value'}],
            data=table_data,
            style_table={
                'width': '100%',
                'maxWidth': '500px',
                'fontFamily': 'Arial, sans-serif',
                'border': ' 1px solid #ddd',
                'borderCollapse': 'collapse',
                'borderRadius': '0px',
                'overflowX': 'auto',
            },
            style_header={
                'backgroundColor': '#f2f2f2',
                'fontWeight': 'bold',
                'textTransform': 'uppercase'
            },
            style_cell={
                'padding': '10px',
                'textAlign': 'left',
                'whiteSpace': 'normal',
                'lineHeight': '1.5'
            },
            style_data={
                'backgroundColor': 'white',
                'color': '#333'
            },
        )
        return html.Div([html.H4(''), table], style={'padding-bottom': '20px'})  # Add a 5px gap below the div
    except Exception as e:
        print(e)
        return None




# The first function 'update_display' is a callback that is triggered when the 'store-conversation' data changes.
@app.callback(
    # This function outputs the 'children' property of the 'display-conversation' component.
    Output("display-conversation", "children"), [Input("store-conversation", "data")]
)
def update_display(chat_history):
    messages = [None]
    # Loop over all chat history data
    for el in chat_history: 
        # Check if the role of the current chat message is 'assistant'
        if el["role"] == "assistant": 
            # If the 'vis' attribute is present in the current message
            if el["vis"]: 
                # Check if the 'vis' attribute is a nested dictionary
                if check_nested_dict(el["vis"]):
                    # If it's a nested dictionary, create several columns
                    messages.append(create_several_columns(el["vis"]))
                else: 
                    # If it's not a nested dictionary, create a single column
                    messages.append(create_single_column(el["vis"]))
            else: 
                # If there's no 'vis' attribute, simply add the content to a textbox
                messages.append(textbox(el["content"], box="AI"))
        # If the role of the current chat message is 'user', add the content to a textbox
        if el["role"] == "user":
            messages.append(textbox(el["content"], box="user"))
    # Return the list of messages
    return messages

# The second function 'generate_graph' is another callback that is triggered when the 'store-conversation' data changes. 
@app.callback(
    # This function outputs the 'data' property of the 'figure-store' component and the 'children' property of the 'my-output' component.
    [Output("figure-store", "data", allow_duplicate=True), Output("my-output", "children")], [Input("store-conversation", "data")],
    # Prevents the callback from being triggered on initial load
    prevent_initial_call=True
)
def generate_graph(chat_history):
    # Get the callback context to understand what input triggered the callback
    ctx = dash.callback_context
    # Extract input id and property which caused the trigger
    input_id = ctx.triggered[0]['prop_id'].split('.')[0]
    print(input_id)
    try: 
        # Generate a graph based on the last question and the code to run in the chat history
        plotly_vis, pandas_code, plotly_code = AgentDF(df, prompt=chat_history[-1]["question"], is_plotly=True, pandas_code=chat_history[-1]["code_to_run"])
        # Format the code in a readable way
        strCode =  ["*Code used for calucalting the values:*", '```python\n' + '\n'.join(pandas_code) + '\n```', "*Code used for the visualization:*",'```python\n' + plotly_code + '\n```']
        # Return the graph and the formatted code
        return [pio.to_json(pio.from_json(plotly_vis))], "\n".join(strCode)
    except Exception as e:
        print("Error", e)
        # If there's an error, return None for the graph and 'ss' for the code
        return [None], "ss"

# Callback function that sets the 'data' property of the 'figure-store' component to "Loading"
# whenever the 'n_clicks' property of the 'submit' component or the 'n_submit' property of the 'user-input' component changes.
@app.callback(
    Output("figure-store", "data", allow_duplicate=True),
    [Input("submit", "n_clicks"), Input("user-input", "n_submit")],
    prevent_initial_call=True
)
def set_loading(n_clicks, n_submit):
    # Return 'Loading'
    return ["Loading"]

# Callback function that clears the 'value' property of the 'user-input' component
# whenever the 'n_clicks' property of the 'submit' component or the 'n_submit' property of the 'user-input' component changes.
@app.callback(
    Output("user-input", "value"),
    [Input("submit", "n_clicks"), Input("user-input", "n_submit")],
)
def clear_input(n_clicks, n_submit):
    # Return empty string
    return ""

# Callback function that updates the 'data' property of the 'store-conversation' component 
# and the 'children' property of the 'loading-component' component whenever the 'n_clicks' property of the 'submit' 
# component or the 'n_submit' property of the 'user-input' component changes.
@app.callback(
    [Output("store-conversation", "data"), Output("loading-component", "children")],
    [Input("submit", "n_clicks"), Input("user-input", "n_submit")],
    [State("user-input", "value"), State("store-conversation", "data")],
    prevent_initial_call=True
)
def run_chatbot(n_clicks, n_submit, user_input, chat_history):
    # If there were no clicks and no submits, return empty data and None
    if n_clicks == 0 and n_submit is None:
        return [], None
    # If the user input is None or empty, return the current chat history and None
    if user_input is None or user_input == "":
        return chat_history, None
    # If the chat history is empty, initialize it with a system message
    if not len(chat_history):
        chat_history = [{"role": "system", "content": '''You are a helpful assistant. You '''}]
    # Append the user's input to the chat history
    chat_history = append_element(chat_history, {"role": "user", "content": user_input})
    # Generate a response from the agent based on the user input
    answer, vis, code_to_run = AgentDF(df, prompt=user_input, is_plotly=False)
    # Append the agent's response to the chat history
    chat_history = append_element(chat_history,  {"role": "assistant", "content": str(answer), "vis" : vis, "code_to_run" : code_to_run, "question" : user_input})
    # Return the updated chat history and None
    return chat_history, None

# Define the structure of the output graph component
output_graph = [
    dbc.CardHeader("Plotly Express Graph"),
    dbc.CardBody(
        [
            dbc.Tabs(
                [
                    dbc.Tab(
                        label="Graph",
                        tab_id="graph-tab",
                        label_style={"font-weight": "bold"},
                    ),
                    dbc.Tab(
                        label="Data",
                        tab_id="data-tab",
                        label_style={"font-weight": "bold"},
                    ),
                ],
                id="graph-tabs",
                card=True,
                active_tab="graph-tab",
                style={"marginTop": "20px"},
            ),
            dcc.Store(id="figure-store"),
            dcc.Store(id="figure-store-update"),  # Add the dcc.Store component
            dcc.Loading(  # Add the dcc.Loading component
                id="loading-component-graph",
                type="default",
                children=html.Div(id="graph-tab-content", className="p-4", style={"height": "425px"}),
            ),
        ]
    ),
]

# Callback function that updates the 'children' property of the 'graph-tab-content' component 
# based on the 'active_tab' property of the 'graph-tabs' component and the 'data' property of the 'figure-store' component.
@app.callback(
    Output("graph-tab-content", "children"),
    [Input("graph-tabs", "active_tab"), Input("figure-store", "data")]
)
def render_tab_content(active_tab, json_figure):
    # If the active tab is the "Graph" tab
    if active_tab == "graph-tab":
        try: 
            print("json_figure", json_figure)
            # If the data is still loading, wait for a second and check again
            while json_figure[0] == "Loading": 
                time.sleep(1)  
            # Load the data into a Plotly figure
            fig = pio.from_json(json_figure[0])
            # Return a dcc.Graph component with the loaded figure
            return dbc.CardBody([
            dcc.Graph(figure=fig, style={"height": "400px", "marginTop": "-40px"})
        ])
        except Exception as e:
            # If there's an error loading the data, return a default figure
            default_fig = px.line(
            df,
            x="Year",
            y="COGS",
            )

            default_fig.update_layout(
                title={
                'text': "Default Figure: Query the chatbot for a different visualization.",
                'y':0.95, # vertical position in normalized coordinates
                'x':0.5, # horizontal position in normalized coordinates
                'xanchor': 'center', # anchor point of x position
                'yanchor': 'top', # anchor point of y position
                }
            ),
            return dbc.CardBody([
                dcc.Graph(figure=default_fig, style={"height": "425px", "marginTop": "-40px"})
            ])
    # If the active tab is the "Data" tab
    elif active_tab == "data-tab":
        # Return a dash_table.DataTable component with the data from the dataframe 'df'
        return dbc.CardBody([
            dash_table.DataTable(
                id="data-table",
                columns=[{"name": col, "id": col} for col in df.columns],
                data=df.to_dict("records"),
                style_table={"maxHeight": "325px", "overflowY": "scroll"},
            )
        ])
# Define the structure of the explanation card component
explanation_card = [
    dbc.CardHeader("What am I looking at?"),
    dbc.CardBody(
        dcc.Loading(
            type="default",
            children=[dcc.Markdown(id='my-output')]
        )
    ),
]

# Define the structure of the left and right columns of the layout
left_col = [dbc.Card(output_graph), html.Br(), dbc.Card(explanation_card)]
right_col  = dbc.Card([
    dbc.CardHeader("Conversation Interface"),
    dbc.CardBody([
    dcc.Store(id="store-conversation", data=""), 
    dbc.Spinner(html.Div(id="loading-component")),
    conversation,
    controls],
    )
])

# Define the layout of the Dash app
app.layout = dbc.Container(
    [
        Header("SCM Generative AI", app),
        html.Hr(),
        dbc.Row([dbc.Col(left_col, md=7), dbc.Col(right_col, md=5)]),
    ],
    fluid=True,
)




if __name__ == "__main__":
    app.run_server(debug=True)
