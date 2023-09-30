# SCM Generative AI 

Welcome to the  Generative AI repository for dynamically creating dashboards by leveraging GPT-4! 

This project employs the power of artificial intelligence to drive and streamline data analysis processes in various contexts. Our project is built on Dash, a powerful framework for building analytical web applications in pure Python, and uses various callback functions to establish interactivity in the application.

The repository provides an interactive chatbot that allows to query an underlying dataset. The chatbot takes user input, processes it, and based on the input, returns appropriate responses or visualizations. It can execute dynamic Python code in response to user prompts and it has in-built support for Plotly, a Python graphing library, making it capable of producing a wide range of highly interactive graphs.

A major part of this project involves parsing and modifying Python code strings, executing these code strings in a specific environment, and capturing the outputs for further use. There are numerous helper functions for string manipulation, including adding a `.to_json()` method call to a string, removing the last 'print' statement from a string, and enclosing a substring within a longer string with `print()`.

The web application layout is designed using the Dash Bootstrap Components library, providing a responsive, clean and professional look.


## OpenAI API Access

In order to obtain access to the GPT API, you will need to register with openai.com. Once you have the API,  you can find the secret key and export it as an environment variable:
```
export OPENAI_KEY="xxxxxxxxxxx"
```
Where "xxxxxxxxxxx" corresponds to your secret key.

## Instructions

To get started, first clone this repo:


Create a conda env:
```
conda create -n venv 
conda activate venv
```

Or a venv (make sure your `python3` is 3.6+):
```
python3 -m venv venv
source venv/bin/activate  # for Windows, use venv\Scripts\activate.bat
```

Install all the requirements:

```
pip install -r requirements.txt
```

You can now run the app:
```
python3 app.py
```

and visit http://127.0.0.1:8050/.

