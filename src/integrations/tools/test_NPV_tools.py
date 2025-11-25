from langchain_core.tools import tool
import json

@tool
def NPV_tool(agent_inputs: dict):
    """
    Calculates the Net Present Value (NPV) based on the production prediction data.

    Args:
        agent_inputs (dict):
            - file_id (str): Identifier for the production prediction data file (required).
            - discount_rate (float): The discount rate to be applied in the NPV calculation (required).

    Returns:
        - JSON object containing the calculated NPV and a success message, or an error message.
    """
    try:
        # Extract required inputs
        file_id = agent_inputs.get('file_id')
        discount_rate = agent_inputs.get('discount_rate')

        # Placeholder for the actual NPV calculation logic
        # Simulate an NPV calculation with a placeholder value
        npv = 1000000  # Replace this with the actual NPV calculation logic

        # Simulate successful operation
        return json.dumps({
            "file_id": file_id,
            "NPV": npv,
            "message": "NPV calculation complete"
        })

    except Exception as e:
        # Catch any error and return the error message
        return json.dumps({"error": str(e)})
