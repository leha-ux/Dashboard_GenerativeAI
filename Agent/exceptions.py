"""Agent's custom exceptions.

This module contains the implementation of Custom Exceptions.

"""

class MissingPrintStatementError(Exception):
    def __init__(self, message="The last line code does not include a print statement. The final answer has to be captured. The answer can only be captured if you print it. Decide what you want to show and enclose the correct part with a print statement."):
        self.message = message
        super().__init__(self.message)


class MissingPlotlyExport(Exception):
    def __init__(self, message='''The maximum height is 425px. Do not return a pandas dataframe. The final answer has to be captured. Do not include a fig.show() statement!! The answer can only be captured if you parse the plotly figure to a json string (pio.to_json(fig)) and then enclose it by print. Assuming your plotly figure is called fig. The code could look like the following: 
    print(pio.to_json(fig))
    '''):
        self.message = message
        super().__init__(self.message)


class APIKeyNotFoundError(Exception):

    """
    Raised when the API key is not defined/declared.

    Args:
        Exception (Exception): APIKeyNotFoundError
    """


class LLMNotFoundError(Exception):
    """
    Raised when the LLM is not provided.

    Args:
        Exception (Exception): LLMNotFoundError
    """


class NoCodeFoundError(Exception):
    """
    Raised when no code is found in the response.

    Args:
        Exception (Exception): NoCodeFoundError
    """


class MethodNotImplementedError(Exception):
    """
    Raised when a method is not implemented.

    Args:
        Exception (Exception): MethodNotImplementedError
    """


class UnsupportedOpenAIModelError(Exception):
    """
    Raised when an unsupported OpenAI model is used.

    Args:
        Exception (Exception): UnsupportedOpenAIModelError
    """


class BadImportError(Exception):
    """
    Raised when a library not in the whitelist is imported.

    Args:
        Exception (Exception): BadImportError
    """

    def __init__(self, library_name):
        """
        __init__ method of BadImportError Class

        Args:
            library_name (str): Name of the library that is not in the whitelist.
        """
        self.library_name = library_name
        super().__init__(
            f"Generated code includes import of {library_name} which"
            " is not in whitelist."
        )
