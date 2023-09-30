# -*- coding: utf-8 -*-
"""

This module includes the implementation of the   Agent class with methods to run
the GPT models on Pandas dataframes. 

Example:


    ```python
    import pandas as pd
    from Agent import Agent

    # Sample DataFrame
    df = pd.DataFrame({
        "country": ["United States", "United Kingdom", "France", "Germany", "Italy",
        "Spain", "Canada", "Australia", "Japan", "China"],
        "gdp": [19294482071552, 2891615567872, 2411255037952, 3435817336832,
        1745433788416, 1181205135360, 1607402389504, 1490967855104, 4380756541440,
        14631844184064],
        "happiness_index": [6.94, 7.16, 6.66, 7.07, 6.38, 6.4, 7.23, 7.22, 5.87, 5.12]
    })

    # Instantiate GPT
    from Agent.llm.openai import OpenAI
    llm = OpenAI(api_token="YOUR_API_TOKEN")


    answer = Agent(prompt=question, is_plotly=false)
    plotly_vis = Agent(prompt=question, is_plotly=True, pandas_code = answer["pandas_code"])

    ```
"""

#Import dependencies
import sys, os
import ast
import io
import logging
import re
import sys
import uuid
import json
import copy
from contextlib import redirect_stdout
from typing import List, Optional, Union
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
import astor
import pandas as pd
import numpy as np
from .constants import (
    WHITELISTED_BUILTINS,
    WHITELISTED_LIBRARIES,
)
from .exceptions import BadImportError, LLMNotFoundError, MissingPrintStatementError, MissingPlotlyExport
from .helpers._optional import import_dependency
from .helpers.cache import Cache
from .llm.base import LLM
from .llm.langchain import LangchainLLM
from .prompts.correct_error_prompt import CorrectErrorPrompt
from .prompts.correct_error_prompt_plotly import CorrectErrorPromptPlotly
from .prompts.generate_python_code import GeneratePythonCodePrompt
from .prompts.generate_plotly_code import GeneratePlotlyCodePrompt
from .prompts.generate_response import GenerateResponsePrompt



# Function to find a specific sequence in a string using regex.
def _find_sequence(string, sequence):
    pattern = re.compile(sequence)  # Compile the pattern
    match = re.search(pattern, string)  # Search for the pattern in the string
    if match:  # If there's a match
        return True  # Return True
    else:
        return False  # If not, return False

# Function to add '.to_json()' to the last word in a string
def _add_to_json(input_str):
    if not isinstance(input_str, str):  # Check if the input is a string
        return 'Invalid input, please input a string.'

    words = input_str.split()  # Split the input string into words

    if not words:  # Check if the string is empty
        return 'Empty string provided.'

    words[-1] = words[-1] + '.to_json()'  # Add .to_json() to the last word

    output_str = ' '.join(words)  # Join the words back into a string

    return output_str  # Return the output string

# Function to remove the last 'print' statement from a string
def _remove_last_print(input_string):
    pattern = r'(.*)(print)(?![^()]*\))(.*)'  # Regex pattern to match 'print'
    match = re.search(pattern, input_string)  # Search for the pattern in the input string
    if match:  # If there's a match
        return match.group(1) + match.group(3)  # Return the string without 'print'
    return input_string  # If not, return the input string

# Function to check if the last element of a list matches the pattern 'print(...)'
def _check_last_element(lst):
    last_element = lst[-1]  # Get the last element of the list
    pattern = r'print\(.+\)$'  # Regex pattern to match "print(...)"
    match = re.match(pattern, last_element)  # Check if the last element matches the pattern
    return bool(match)  # Return True if there's a match, False otherwise

# Function to enclose a substring within a longer string with 'print()'
def _enclose_with_print(longer_string, substring):
    if substring in longer_string:  # If the substring is in the longer string
        enclosed_string = f"print({substring})"  # Enclose the substring with 'print()'
        result = longer_string.replace(substring, enclosed_string)  # Replace the substring in the longer string with the enclosed string
        return result  # Return the result
    else:
        return "Substring not found in the longer string."  # If the substring is not in the longer string, return an error message

# Function to enclose a string with 'print()'
def _enclose_in_print(string):
    return f"print({string})"  # Return the string enclosed with 'print()'

# Class to parse and manipulate a list of strings
class dfParser:
    # Constructor
    def __init__(self, input_str, lst, environment):
        try:
            self.environment = environment  # Set the environment
            self.input_str = input_str  # Set the input string
            self.lst = lst  # Set the list
            if _check_last_element(self.lst):  # If the last element of the list matches the pattern 'print(...)'
                self.lst[-1] = _enclose_in_print(_add_to_json(_remove_last_print(self.lst[-1])))  # Remove 'print', add '.to_json()', and enclose with 'print()'
            else: 
                self.lst[-1] = _enclose_in_print(_add_to_json(self.lst[-1]))  # Add '.to_json()' to the last element and enclose it with 'print()'
            self.formatted_code = "\n".join(self.lst)  # Join the list into a string with newline characters
            #print("0", self.formatted_code)
        except Exception as e:  # Catch any exceptions
            print(e)  # Print the exception

    # Method to execute the formatted code and read the output
    def read_output(self):
        try: 
            with redirect_stdout(io.StringIO()) as output:  # Redirect the standard output to a string
                exec(self.formatted_code, self.environment)  # Execute the formatted code in the given environment
            return str(output.getvalue())  # Return the output as a string
        except Exception as e:  # Catch any exceptions
            return None  # Return None if there's an exception


class Agent:
    """
    Agent is a wrapper around a LLM to make dataframes conversational and visualize a plotly visualization.


    This is an entry point of `Agent` object. This class consists of methods
    to interface the LLMs with Pandas dataframes returning a numeric answer and a plotly visualization. A pandas dataframe metadata i.e.
    df.head() and prompt is passed on to GPT to generate a Python
    code to answer the questions asked. The resultant python code is run on actual data
    and answer is converted into a conversational form and returned as a plotly visualization.

    Args:
        _llm (obj): LLMs option to be used for API access
        _verbose (bool, optional): To show the intermediate outputs e.g. python code
        generated and execution step on the prompt. Default to False
        _is_conversational_answer (bool, optional): Whether to return answer in
        conversational form. Default to False
        _enforce_privacy (bool, optional): Do not display the data on prompt in case of
        Sensitive data. Default to False
        _max_retries (int, optional): max no. of tries to generate code on failure.
        Default to 3
        _original_instructions (dict, optional): The dict of instruction to run. Default
        to None
        last_code_generated (str, optional): Pass last Code if generated. Default to
        None
        last_code_executed (str, optional): Pass the last execution / run. Default to
        None
        code_output (str, optional): The code output if any. Default to None
        last_error (str, optional): Error of running code last time. Default to None
        prompt_id (str, optional): Unique ID to differentiate calls. Default to None


    Returns (str): Response to a Question related to Data

    """

    _llm: LLM
    _verbose: bool = False
    _is_conversational_answer: bool = False
    _enforce_privacy: bool = False
    _max_retries: int = 3
    _original_instructions: dict = {
        "question": None,
        "df_head": None,
        "num_rows": None,
        "num_columns": None,
    }
    _cache: Cache = Cache("chat")
    _cache_plotly: Cache = Cache("plotly")
    _enable_cache: bool = True
    _prompt_id: Optional[str] = None
    _additional_dependencies: List[dict] = []
    _custom_whitelisted_dependencies: List[str] = []
    last_code_generated: Optional[str] = None
    last_plotly_code_generated: Optional[str] = None
    last_code_executed: Optional[str] = None
    code_output: Optional[str] = None
    plotly_code_output: Optional[str] = None
    last_error: Optional[str] = None

    def __init__(
        self,
        llm=None,
        conversational=False,
        verbose=False,
        enforce_privacy=False,
        enable_cache=True,
        custom_whitelisted_dependencies=None,
        pandas_code = None,
    ):
        """

        __init__ method of the Class Agent

        Args:
            llm (object): ChatGPT object. Default is None
            conversational (bool): Whether to return answer in conversational form.
            Default to False
            verbose (bool): To show the intermediate outputs e.g. python code
            generated and execution step on the prompt.  Default to False
            enable_cache (bool): Enable the cache to store the results.
            Default to True
            custom_whitelisted_dependencies (list): List of custom dependencies to
            be used. Default to None
        """

        handlers = [logging.FileHandler("Agent.log")]
        if verbose:
            handlers.append(logging.StreamHandler(sys.stdout))
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            handlers=handlers,
        )
        self._logger = logging.getLogger(__name__)

        if llm is None:
            raise LLMNotFoundError(
                "An LLM should be provided to instantiate a Agent instance"
            )
        self._load_llm(llm)
        self._is_conversational_answer = conversational
        self._verbose = verbose
        self._enforce_privacy = enforce_privacy
        self._enable_cache = enable_cache
        self._process_id = str(uuid.uuid4())
        self.pandas_code = pandas_code


        if custom_whitelisted_dependencies is not None:
            self._custom_whitelisted_dependencies = custom_whitelisted_dependencies

    def _load_llm(self, llm):
        """
        Check if it is a Agent LLM or a Langchain LLM.
        If it is a Langchain LLM, wrap it in a Agent LLM.

        Args:
            llm (object): LLMs option to be used for API access

        Raises:
            BadImportError: If the LLM is a Langchain LLM but the langchain package
            is not installed
        """

        try:
            llm.is_Agent_llm()
        except AttributeError:
            llm = LangchainLLM(llm)

        self._llm = llm

    def conversational_answer(self, question: str, answer: str) -> str:
        """
        Returns the answer in conversational form about the resultant data.

        Args:
            question (str): A question in Conversational form
            answer (str): A summary / resultant Data

        Returns (str): Response

        """

        if self._enforce_privacy:
            # we don't want to send potentially sensitive data to the LLM server
            # if the user has set enforce_privacy to True
            return answer

        instruction = GenerateResponsePrompt(question=question, answer=answer)
        return self._llm.call(instruction, "")


    def create_plotly(
        self,
        data_frame: Union[pd.DataFrame, List[pd.DataFrame]],
        prompt: str,
        pandas_code: str,
        is_conversational_answer: bool = True,
        show_code: bool = True,
        anonymize_df: bool = False,
        use_error_correction_framework: bool = True,
    ) -> Union[str, pd.DataFrame]:
        """
        Run the Agent to create a visualization for the question and the given pandas dataframe.

        Args:
            data_frame (Union[pd.DataFrame, List[pd.DataFrame]]): A pandas Dataframe
            prompt (str): A prompt to query about the Dataframe
            use_error_correction_framework (bool):  Error Correction mechanism.

        Returns (str): A plotly visualization printed as html

        """

        self.log(f"Running Agent for plotly with {self._llm.type} LLM...")

        self._prompt_id = str(uuid.uuid4())
        self.log(f"Plotly Prompt ID: {self._prompt_id}")

        try:


            rows_to_display = 0 if self._enforce_privacy else 5


            df_head = data_frame.head(rows_to_display)

            code = self._llm.generate_code(
                GeneratePlotlyCodePrompt(
                    question=prompt,
                    df_head=df_head,
                    num_rows=data_frame.shape[0],
                    num_columns=data_frame.shape[1],
                    pandas_code = pandas_code
                ),
                prompt,
            )

            self.last_plotly_code_generated = code
            self.log(
                f"""
                    Plotly Code generated:
                    ```
                    {code}
                    ```
                """
            )
                

            code_to_run, last_line, lines, environment, captured_output = self.run_code(
                code,
                data_frame,
                is_plotly = True,
                output_error = MissingPlotlyExport(),
                show_errors=False,
                use_error_correction_framework=use_error_correction_framework,
            )

        
            #code_to_run = _enclose_with_print(code_to_run,lines[-1])
            with redirect_stdout(io.StringIO()) as output:
                exec(("\n".join(code_to_run)), environment)
                plotly_answer = output.getvalue()
            #print("Answer to be shown", output.getvalue())
            self.plotly_code_output = plotly_answer
            self.log(f"Answer Plotly: {plotly_answer}")


            return (plotly_answer, pandas_code, "\n".join(code_to_run))
        except Exception as exception:
            self.last_error = str(exception)
            print("Exception", exception)
            exc_type, exc_obj, exc_tb = sys.exc_info()
            print(exc_type, exc_tb.tb_lineno)
            return None


    def run(
        self,
        data_frame: Union[pd.DataFrame, List[pd.DataFrame]],
        prompt: str,
        is_conversational_answer: bool = True,
        show_code: bool = True,
        anonymize_df: bool = False,
        use_error_correction_framework: bool = True,
    ) -> Union[str, pd.DataFrame, str]:
        """
        Run the Agent to make Dataframes Conversational.

        Args:
            data_frame (Union[pd.DataFrame, List[pd.DataFrame]]): A pandas Dataframe
            prompt (str): A prompt to query about the Dataframe
            use_error_correction_framework (bool):  Error Correction mechanism.

        Returns (str): Answer to the Input Questions about the DataFrame and the code used for visualization in the frontend 

        """

        self.log(f"Running Pandas Agent with {self._llm.type} LLM...")

        self._prompt_id = str(uuid.uuid4())
        self.log(f"Pandas Prompt ID: {self._prompt_id}")

        try:
            if self._enable_cache and self._cache.get(prompt):
                self.log("Pandas: Using cached response")
                code = self._cache.get(prompt)
            else:
                rows_to_display = 0 if self._enforce_privacy else 5

                multiple: bool = isinstance(data_frame, list)
                df_head = data_frame.head(rows_to_display)

                code = self._llm.generate_code(
                    GeneratePythonCodePrompt(
                        prompt=prompt,
                        df_head=df_head,
                        num_rows=data_frame.shape[0],
                        num_columns=data_frame.shape[1],
                    ),
                    prompt,
                )

                self._original_instructions = {
                    "question": prompt,
                    "df_head": df_head,
                    "num_rows": data_frame.shape[0],
                    "num_columns": data_frame.shape[1],
                }

                self.last_code_generated = code
                self.log(
                    f"""
                        Pandas
                        Code generated:
                        ```
                        {code}
                        ```
                    """
                )

                self._cache.set(prompt, code)


            code_to_run, last_line, lines, environment, captured_output = self.run_code(
                code,
                data_frame,
                is_plotly = False,
                output_error = MissingPrintStatementError(),
                use_error_correction_framework=use_error_correction_framework,
            )
            
            #Darstellung / Error handling 
            if dfParser("\n".join(code_to_run), copy.deepcopy(lines), (environment)):
                vis = dfParser("\n".join(code_to_run), copy.deepcopy(lines), environment).read_output()
            else: 
                vis = None 
            #print("VIS", vis)
            if len(lines) > 1:
                if not _find_sequence(lines[-1], "print"):
                    lines[-1] = _enclose_in_print(lines[-1])
                #code_to_run = _enclose_with_print(code_to_run,lines[-1])
                with redirect_stdout(io.StringIO()) as output:
                    exec(("\n".join(code_to_run)), environment)
                    answer = output.getvalue()
                #print("Answer to be shown", output.getvalue())
            else: 
                answer =  eval(last_line, environment)
            self.code_output = answer
            self.log(f"Pandas Answer: {answer}")

        #Generate plotly code 

            if is_conversational_answer is None:
                is_conversational_answer = self._is_conversational_answer
            if is_conversational_answer:
                answer = self.conversational_answer(prompt, answer)
                self.log(f" Pandas Conversational answer: {answer}")
            return answer, vis, code_to_run
        except Exception as exception:
            self.last_error = str(exception)
            print(exception)
            exc_type, exc_obj, exc_tb = sys.exc_info()
            print(exc_type, exc_tb.tb_lineno)
            return (
                "Unfortunately, I was not able to answer your question, "
                "because of the following error:\n"
                f"\n{exception}\n"
            ), None, None



    def clear_cache(self):
        """
        Clears the cache of the Agent instance.
        """
        self._cache.clear()

    def __call__(
        self,
        data_frame: Union[pd.DataFrame, List[pd.DataFrame]],
        prompt: str,
        is_plotly: bool,
        is_conversational_answer: bool = None,
        show_code: bool = False,
        anonymize_df: bool = True,
        use_error_correction_framework: bool = True,

        pandas_code: str = None
    ) -> Union[str, pd.DataFrame]:
        """
        __call__ method of Agent class. It calls the `run` method.

        Args:
            data_frame:
            prompt:
            is_conversational_answer:
            show_code:
            anonymize_df:
            use_error_correction_framework:

        Returns (str): Answer to the Input Questions about the DataFrame.

        """
        if not is_plotly: 
            return self.run(
                data_frame,
                prompt,
                is_conversational_answer,
                show_code,
                anonymize_df,
                use_error_correction_framework,
            )
        else: 
            return self.create_plotly(
                data_frame,
                prompt,
                pandas_code,
                is_conversational_answer,
                show_code,
                anonymize_df,
                use_error_correction_framework
            )

    def _check_imports(self, node: Union[ast.Import, ast.ImportFrom]):
        """
        Add whitelisted imports to _additional_dependencies.

        Args:
            node (object): ast.Import or ast.ImportFrom

        Raises:
            BadImportError: If the import is not whitelisted

        """
        # clear recent optional dependencies
        self._additional_dependencies = []

        if isinstance(node, ast.Import):
            module = node.names[0].name
        else:
            module = node.module

        library = module.split(".")[0]

        if library == "pandas":
            return

        if library in WHITELISTED_LIBRARIES + self._custom_whitelisted_dependencies:
            for alias in node.names:
                self._additional_dependencies.append(
                    {
                        "module": module,
                        "name": alias.name,
                        "alias": alias.asname or alias.name,
                    }
                )
            return

        if library not in WHITELISTED_BUILTINS:
            raise BadImportError(library)

    def _is_df_overwrite(self, node: ast.stmt) -> bool:
        """
        Remove df declarations from the code to prevent malicious code execution.

        Args:
            node (object): ast.stmt

        Returns (bool):

        """

        return (
            isinstance(node, ast.Assign)
            and isinstance(node.targets[0], ast.Name)
            and re.match(r"df\d{0,2}$", node.targets[0].id)
        )

    def _clean_code(self, code: str) -> str:
        """
        A method to clean the code to prevent malicious code execution

        Args:
            code(str): A python code

        Returns (str): Returns a Clean Code String

        """

        tree = ast.parse(code)

        new_body = []

        for node in tree.body:
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                self._check_imports(node)
                continue
            if self._is_df_overwrite(node):
                continue
            new_body.append(node)

        new_tree = ast.Module(body=new_body)
        return astor.to_source(new_tree).strip()

    def run_code(
        self,
        code: str,
        data_frame: pd.DataFrame,
        output_error: Exception,
        is_plotly: bool,
        use_error_correction_framework: bool = True,
        show_errors = False
    ) -> str:
        """
        A method to execute the python code generated by LLMs to answer the question
        about the input dataframe. Run the code in the current context and return the
        result.

        Args:
            code (str): A python code to execute
            data_frame (pd.DataFrame): A full Pandas DataFrame
            use_error_correction_framework (bool): Turn on Error Correction mechanism.
            Default to True

        Returns (str): String representation of the result of the code execution.

        """

        multiple: bool = isinstance(data_frame, list)


        # Get the code to run removing unsafe imports and df overwrites
        code_to_run = self._clean_code(code)
        self.last_code_executed = code_to_run
        if (is_plotly):
            self.log(
                f"""
            Plotly Code running:
            ```
            {code_to_run}
            ```"""
            )
        else:
            self.log(
                f"""
            Pandas Code running:
            ```
            {code_to_run}
            ```"""
            )

        environment: dict = {
            "pd": pd,
            "np" : np,
            "go" : go,
            "pio" : pio,
            **{
                lib["alias"]: getattr(import_dependency(lib["module"]), lib["name"])
                if hasattr(import_dependency(lib["module"]), lib["name"])
                else import_dependency(lib["module"])
                for lib in self._additional_dependencies
            },
            "__builtins__": {
            "__import__": __import__,
            "px": px,
            "pio" : pio,
                **{builtin: __builtins__[builtin] for builtin in WHITELISTED_BUILTINS},
            },
        }


        environment["df"] = data_frame

        # Redirect standard output to a StringIO buffer
        with redirect_stdout(io.StringIO()) as output:
            count = 0
            while count < self._max_retries:
                try:
                    if (is_plotly):
                        # Regular expression pattern to match ".show()"
                        pattern = r"\.show\(\)"
                        # Search for the pattern in the code string
                        match = re.search(pattern, code_to_run)
                        # Check if the pattern is found
                        if match:
                            raise output_error
                    # Execute the code
                    exec(code_to_run, environment)
                    code = code_to_run
                    if output.getvalue():
                        break
                    else: 
                        raise output_error
                except Exception as e:
                    if not use_error_correction_framework:
                        raise e

                    count += 1

                    if(show_errors):
                        print("we are in:", count)
                        print("We encountered the error", e)
                    
                    if not is_plotly:
                        error_correcting_instruction = CorrectErrorPrompt(
                            code=code,
                            error_returned=e,
                            question=self._original_instructions["question"],
                            df_head=self._original_instructions["df_head"],
                            num_rows=self._original_instructions["num_rows"],
                            num_columns=self._original_instructions["num_columns"],
                        )
                    else: 
                        error_correcting_instruction = CorrectErrorPromptPlotly(
                            code=code,
                            error_returned=e,
                            pandas_code = self.pandas_code,
                            question=self._original_instructions["question"],
                            df_head=self._original_instructions["df_head"],
                            num_rows=self._original_instructions["num_rows"],
                            num_columns=self._original_instructions["num_columns"],
                        )                   
                    code_to_run = self._llm.generate_code(
                        error_correcting_instruction, ""
                    )




        captured_output = output.getvalue()


        # Evaluate the last line and return its value or the captured output
        lines = code.strip().split("\n")

        if(show_errors):
            print("Captured Output", captured_output)
            print("LINES::::", lines)
            print("Code to run::::", code_to_run)

        last_line = lines[-1].strip()

        match = re.match(r"^print\((.*)\)$", last_line)
        if match:
            last_line = match.group(1)

            #return lines, last_line, environment 
        return code_to_run.strip().split("\n"), last_line, lines, environment, captured_output

    def log(self, message: str):
        """Log a message"""
        self._logger.info(message)

    def process_id(self) -> str:
        """Return the id of this Agent object."""
        return self._process_id

    def last_prompt_id(self) -> str:
        """Return the id of the last prompt that was run."""
        if self._prompt_id is None:
            raise ValueError("Pandas AI has not been run yet.")
        return self._prompt_id
