import json
import os
from typing import Annotated
import uuid
from langchain_core.tools import tool
from langchain_experimental.utilities import PythonREPL
from matplotlib import pyplot as plt
repl = PythonREPL()
@tool
def python_repl(
    code: Annotated[str, "The python code to execute to generate your chart."],
):
    """
    You are an expert python code generator, specialized in generating Python code to create a plot.
    Prompt the user before calling this tool. First rule: Only generate the python code, without any additional explanation.
    If you want to see the output of a value, you should print it out with `print(...)`. This is visible to the user."""
    try:
        result = repl.run(code)
        IMAGE_DIR = "src/static/images"  # Make sure this directory exists and is served by FastAPI
        file_name = f"{uuid.uuid4().hex[:6]}.png"
        file_path = os.path.join(IMAGE_DIR, file_name)
        plt.savefig(file_path)
        plt.close()

        # Construct the URL to access the saved image
        image_url = f"http://localhost:800/static/images/{file_name}"
        print(f"Generated image URL: {image_url}")
        result_str = f"Successfully executed:\n```python\n{code}\n```\nStdout: {result}"
        # Return the URL in JSON format
        return json.dumps({
            "image_url": f"Image URL: {image_url}",
            "messages": result_str
        })
    except BaseException as e:
        return f"Failed to execute. Error: {repr(e)}"