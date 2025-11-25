from langchain_core.tools import tool
import json
import uuid

import uuid
import json

@tool
def DL_pred_pretrain(agent_inputs: dict):
    """
    Pretrained deep learning model used for production forecasting.
    available_plays = ['Montney', 'Duvernay', 'Bakken', 'Deep Basin']
    Only the wells with a pretrained model can be used for prediction.
    The prediction is based on geological data (required) and optionally includes historical production data.

    Args:
        agent_inputs (dict): 
            - 'play_name': The name of the formation with a pretrained model (required).
            - 'file_id' (str): Identifier for the input geological data file (required).
            - 'prod_hist' (list): Production history data (optional). Defaults to None or an empty list if not provided.
            - 'shut_list' (list): Shut-in periods data (optional). Defaults to None or an empty list if not provided.

    Output:
        - JSON object containing a new 'file_id' for the predicted file and a 'message' indicating success, 
          or an error message if the formation is not available.
    """

    # List of available pretrained models
    available_plays = ['Montney', 'Duvernay', 'Bakken', 'Deep Basin']

    try:
        # Extract 'formation_name' from inputs
        play_name = agent_inputs.get('play_name')

        # Check if the provided formation name has a pretrained model
        if play_name not in available_plays:
            return json.dumps({
                "file_id": None,
                "message": f"Error: Pretrained model for formation '{play_name}' not available."
            })

        # Generate a new file_id for the prediction result (simulating the file creation)
        new_file_id = f"predicted_file_{uuid.uuid4()}"

        # Simulate a successful operation
        output_results = {
            "file_id": new_file_id,  # New file ID generated for the prediction result
            "message": "Prediction complete"
        }

        return json.dumps(output_results)

    except Exception as e:
        # Return error message in case of failure
        return json.dumps({"file_id": None, "message": str(e)})

    