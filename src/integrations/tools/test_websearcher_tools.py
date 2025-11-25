from langchain_core.tools import tool
from typing import Annotated, List, Optional, Dict
import uuid
import json

IMAGE_DIR = "src/static/images"

@tool
def websearch_tool(
    query: Annotated[str, "The search query provided by the user."]
) -> Annotated[str, "A message containing the search results or an error message."]:
    """
    Simulates a web search based on the user's query.

    Args:
        query (str): The search query provided by the user.

    Returns:
        - JSON object containing a list of simulated web search results (URLs and descriptions) or an error message.
    """
    try:
        # Simulate a web search by generating mock results
        if not query:
            raise ValueError("Search query is missing or invalid.")
        
        # Example mock results (replace with actual search results in real implementation)
        search_results = [
            {
                "title": "Example result 1",
                "url": f"https://example.com/result1?query={uuid.uuid4()}",
                "description": "This is a description of the first search result."
            },
            {
                "title": "Example result 2",
                "url": f"https://example.com/result2?query={uuid.uuid4()}",
                "description": "This is a description of the second search result."
            },
            {
                "title": "Example result 3",
                "url": f"https://example.com/result3?query={uuid.uuid4()}",
                "description": "This is a description of the third search result."
            }
        ]

        # Simulate successful search
        return json.dumps({
            "results": search_results,
            "message": "Search completed successfully."
        })

    except Exception as e:
        # Handle any exceptions and return an error message
        return json.dumps({"error": str(e)})

