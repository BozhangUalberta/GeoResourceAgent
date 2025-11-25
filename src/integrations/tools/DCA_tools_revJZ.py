# import sys
# import os
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.integrations.APIfunctions.DCA_API_revJZ import ProcessShutSeqs
import json
import numpy as np
from langchain_core.tools import tool
from typing import Annotated, List, Optional
from langgraph.prebuilt import InjectedState
import os
import sqlite3
import pandas as pd
from src.utils.db_utils import store_data_sqlite3
import uuid

def load_data_to_dataframe(db_name, table_name):
    # Load the entire table into a DataFrame
    conn = sqlite3.connect(db_name)
    df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
    conn.close()
    return df

@tool
def DCA_tool_table_source(
    userID: Annotated[str, InjectedState("userID")],
    table_name: Annotated[str, "The table containing the user's required production data"],
    column_name: Annotated[str, "The column for the desired feature to plot"],
    well_identifier_col_name: Annotated[str, "Column name for well identifier"],
    extend_length: Annotated[Optional[int], "The length by which to extend the forecasted production profile."] = None,
    well_names: Annotated[Optional[List[str]], "Wells to do DCA"] = None,
    All_wells: Annotated[Optional[bool], "Do DCA for all the wells in the file"] = False,
    shut_signal: Annotated[Optional[List[float]], "The signal data for shut-in periods, if applicable."] = None,
    peak_t: Annotated[Optional[int], "The time at which peak production occurs."] = None,
    shut_threshold: Annotated[Optional[float], "Below this rate, the well is considered shut-in."] = 30,
    screen_ratio: Annotated[Optional[float], "Rapid decline threshold. For example, if qt+1/qt < screen_ratio, it will be considered as shut-in."] = 0.1
) -> Annotated[str, "JSON containing the fitted curve, initial decline rate (Di), and b-factor."]:
    """
    Performs Decline Curve Analysis (DCA) using production data sourced from a database table.
    Key Considerations:
    - This DCA tool analyzes data for multiple wells but processes each well individually.
    - Ensure that if the table contains multiple wells, the user specifies which wells to predict.
    - Use the appropriate SQL processing tools to handle data retrieval and manipulation.
    - You need to let the user specific which production data to perform DCA: gas, oil or water.
    - You MUST show the next step message to the user.
    """

    print('========== DCA table source, multiple wells ===========')
    print(well_identifier_col_name)
    print(column_name)

    try:
        if extend_length is None:
            extend_length = 0

        db_name = os.path.join("database", userID + ".db")
        df = load_data_to_dataframe(db_name, table_name)

        # Filter Data
        if not All_wells:
            if well_names:
                df = df[df[well_identifier_col_name].isin(well_names)]
                print(well_names)
            else:
                return json.dumps({
                    "message": "Please specify the well names or set All_wells to True."
                })

        if df.empty:
            return json.dumps({
                "message": "No data available for the specified wells."
            })

        # Ensure the specified column is numeric
        df[column_name] = pd.to_numeric(df[column_name], errors='coerce')

        # Drop rows with NaN values in the specified column
        df = df.dropna(subset=[column_name])

        if df.empty:
            return json.dumps({
                "message": "No valid data after removing null values."
            })

        # Initialize the result DataFrame
        fit_df = pd.DataFrame()

        # Group the DataFrame by well identifier
        grouped_df = df.groupby(well_identifier_col_name)

        # Process each well individually
        for well_name, group in grouped_df:
            well_data = group.copy()

            # Interpolate null values
            well_data[column_name] = well_data[column_name].interpolate(method='linear', limit_direction='forward')

            # Drop remaining NaN values
            well_data = well_data.dropna(subset=[column_name])

            if well_data.empty:
                continue  # Skip wells with no valid data

            # Convert the column to a list of floats
            input_seq = well_data[column_name].tolist()

            # Initialize DCA
            # dca = ProcessShutSeqs(
            #     shut_threshold=shut_threshold,
            #     screen_ratio=screen_ratio,
            #     shut_signal=shut_signal,
            #     peak_t=peak_t
            # )
            dca = ProcessShutSeqs(raw_data = input_seq, fit_equation='hyperbolic', extend_length=extend_length)

            # Perform DCA
            dca.find_peak()
            dca.remove_downtime()
            dca.find_fit_params()
            dca.generate_decline_curve()
            dca.adjust_peak()
            dca.add_terminal_decline()
            dca.add_back_downtime()
            dca.add_back_pre_peak()
            dca.prepare_final_output()

            q_fit = dca.fit_data_output
            qi_s, b_ss, Di_ss = dca.fit_params
            b_s = [b_ss]*len(q_fit)
            Di_s = [Di_ss]*len(q_fit)
            

            # Prepare the results DataFrame for this well
            result_length = len(q_fit)
            original_length = len(input_seq)

            # Extend well_data to match the length of q_fit
            if result_length > original_length:
                additional_rows = result_length - original_length
                last_date = well_data.index[-1]
                additional_dates = pd.date_range(start=last_date, periods=additional_rows+1, freq='D')[1:]
                additional_df = pd.DataFrame({
                    well_identifier_col_name: [well_name] * additional_rows,
                    column_name: [np.nan] * additional_rows
                }, index=additional_dates)
                well_data = pd.concat([well_data, additional_df])

            # Add the DCA results to well_data
            q_fit_col_name = 'q_fit_' + column_name
            well_data[q_fit_col_name] = q_fit
            well_data['Di'] = Di_s
            well_data['b'] = b_s

            # Append the results to the fit_df DataFrame
            fit_df = pd.concat([fit_df, well_data], ignore_index=False)

        if fit_df.empty:
            return json.dumps({
                "message": "DCA could not be performed on any wells due to insufficient data."
            })

        # Store the results back to the database
        conn = sqlite3.connect(db_name)
        new_table_name = table_name + '_DCA_results_' + uuid.uuid4().hex[:4]
        fit_df.to_sql(new_table_name, conn, if_exists='replace', index=False)
        # Prepare a descriptive entry for the agent
        if well_names:
            description = (
                f"Processed DCA results for the specified wells: {', '.join(well_names)}. "
                f"The results include decline rate (Di), b-factor, and fitted production values stored as '{q_fit_col_name}' in the table '{new_table_name}'."
            )
        else:
            description = (
                f"Processed DCA results including decline rate (Di), b-factor, and fitted production values stored as '{q_fit_col_name}' in the table '{new_table_name}'."
            )

        # Log the saved table for the agent
        store_data_sqlite3(
            filename=userID,
            table="running",
            data=f"Saved table: {new_table_name}",
            type="processed",
            description=description
        )

        conn.commit()
        conn.close()

        # Prepare output to return to the user
        grouped_results = fit_df.groupby(well_identifier_col_name).apply(
            lambda group: {
                "Di": group['Di'].iloc[-1],
                "b": group['b'].iloc[-1]
            }
        ).to_dict()

        total_wells = len(grouped_results)

        next_step_message = (
            f"Ask if the user wants to visualize the fitted curves in the stored table '{new_table_name}', which contains the DCA results."
        )
        potential_plot_prompt = (
            "Plot the original rate in gray and the fitted curve in light blue (alpha = 0.7). "
            "Use the legend to show only two groups: 'Original rate' and 'DCA fitted', instead of listing individual wells."
            "Align all the production profiles to the same start."
        )

        if total_wells > 20:
            # Get the first 20 wells
            limited_results = dict(list(grouped_results.items())[:20])
            message = (
                f"DCA successful. The fitted time series is stored in the database table '{new_table_name}' "
                f"with the column '{q_fit_col_name}'.\n"
                f"Displaying results for the first 20 wells out of {total_wells}. "
                f"{total_wells - 20} wells are hidden. Check the saved table '{new_table_name}' for detailed results."
            )
            DCA_results = limited_results
        else:
            # Display all results if there are 20 or fewer wells
            message = (
                f"DCA successful. The fitted time series is stored in the database table '{new_table_name}' "
                f"with the column '{q_fit_col_name}'."
            )
            DCA_results = grouped_results

        # Combine shared keys into a single return dictionary
        output_results = {
            "message": message,
            "DCA_results": DCA_results,
            "next_step": (
                f"{next_step_message}, with the plotting prompt:{potential_plot_prompt}"
            )
        }

        return json.dumps(output_results)

    except Exception as e:
        # Catch any error and return the error message
        print(str(e))
        return json.dumps({"error": str(e)})


@tool
def DCA_tool_conversation_source(
    input_seq: Annotated[List[float], "The input sequence for the production profile"],
    extend_length: Annotated[Optional[int], "The length by which to extend the forecasted production profile."] = None,
    shut_signal: Annotated[Optional[List[float]], "The signal data for shut-in periods, if applicable."] = None,
    peak_t: Annotated[Optional[int], "The time at which peak production occurs."] = None
) -> Annotated[str, "JSON containing the fitted curve, initial decline rate (Di), and b-factor."]:
    """
    Performs Decline Curve Analysis (DCA) to fit a production profile using data provided directly in the conversation.
    This tool is suitable for short input sequences that are fron the conversation.
    
    ### Parameters
    - **input_seq** (list of floats): The production data sequence to analyze, provided directly through conversation.
    - **extend_length** (int, optional): Number of additional data points to extend the forecasted profile. Defaults to 0 if not provided.
    - **shut_signal** (list of floats, optional): Data representing shut-in periods, if applicable.
    - **peak_t** (int, optional): The time point at which peak production occurs.
    
    ### Returns
    - A JSON string with fitted curve, stored table name, Di and b.
    """
    print('========== DCA conversation source ===========')
    
    try:
        print('Performing DCA')
        
        # Set default for extend_length if not provided
        extend_length = extend_length or 0
        
        # Ensure the input sequence is in float format
        input_seq = [float(item) for item in input_seq]

        # Perform DCA using ProcessShutSeqs
        dca = ProcessShutSeqs(shut_threshold=50, screen_ratio=0.5, shut_signal=shut_signal, peak_t=peak_t)

        # Perform DCA calculation
        q_fit, Di_s, b_s = dca.complex_arps(input_seq, extend_length)

        # Prepare the output in JSON format
        output_results = {
            "Fitted curve": q_fit.tolist(),
            "Di": np.unique(Di_s)[-1].tolist(),
            "b": np.unique(b_s)[-1].tolist()
        }
        return json.dumps(output_results)

    except Exception as e:
        # Catch and return error message
        print(f"Error occurred: {e}")
        return json.dumps({"error": str(e)})