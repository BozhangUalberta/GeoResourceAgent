from langchain.tools import tool
import numpy as np
import numpy_financial as npf
from typing import Dict,Optional, Union, List
from src.utils.logger_utils import get_logger
from typing import Annotated
from langgraph.prebuilt import InjectedState
import pandas as pd
import uuid
import sqlite3
from src.utils.db_utils import store_data_sqlite3
import os
import json

## Load SQL table data to df
def load_data_to_dataframe(db_name, table_name):
    # Load the entire table into a DataFrame
    conn = sqlite3.connect(db_name)
    df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
    conn.close()
    return df

# # Extract optional data
# def extract_or_default(df, col_name, default=[0]):
#     return df[col_name].values if col_name in df.columns else default

#############################
### Calculate well c_star ###
#############################

@tool
def calculate_well_c_star(
    userID: Annotated[str, InjectedState("userID")],
    TVDmax: Annotated[float, "Deepest True Vertical Depth (m)"],
    TLL: Annotated[float, "Total Lateral Length (m)"],
    TPP: Annotated[float, "Total Equivalent Proppant Placed (tonnes)"],
    TMD: Annotated[float, "Total Measured Depth (m)"],
    TVDavg: Annotated[Optional[float], "Average True Vertical Depth (m), defaults to TVDmax if not provided"] = None,
    ACCI: Annotated[Optional[float], "Alberta Capital Cost Index, defaults to 1.0"] = 1.0,
) -> Annotated[Dict[str, Union[str, float, Dict]], "Calculation results and metadata"]:
    """
    Calculate well C* (C*) based on drilling and completion parameters.
    
    Returns:
    - A dictionary containing calculation results and metadata.
    """
    logger = get_logger(userID )
    logger.info(f"Calculate well C*....<br>")
    print("Calculate well C*....")
    try:
        # Set default values and handle edge cases
        TVDmax = max(TVDmax, 249)
        TVDavg = TVDavg if TVDavg is not None else TVDmax

        # Calculate Y factor (multi-leg well cost adjustment)
        def calculate_y_factor(TMD: float, TVDavg: float) -> float:
            if TMD / TVDavg < 10:
                return 1.0
            return max(0.24, 1.39 - 0.04 * (TMD / TVDavg))

        Y = calculate_y_factor(TMD, TVDavg)

        # Calculate capital cost based on TVDmax
        base_cost = 1170 * (TVDmax - 249)
        depth_adjustment = 3120 * (TVDmax - 2000) if TVDmax > 2000 else 0
        lateral_cost = Y * 800 * TLL
        completion_cost = 0.6 * TVDavg * TPP

        c_star = ACCI * (
            base_cost +
            depth_adjustment +
            lateral_cost +
            completion_cost
        )

        return {
            "status": "success",
            "data": {
                "c_star": c_star,
                "formatted_cost": f"${c_star:,.2f}"
            },
            "metadata": {
                "y_factor": Y,
                "base_cost": base_cost,
                "depth_adjustment": depth_adjustment,
                "lateral_cost": lateral_cost,
                "completion_cost": completion_cost,
                "inputs": {
                    "TVDmax": TVDmax,
                    "TVDavg": TVDavg,
                    "TLL": TLL,
                    "TPP": TPP,
                    "TMD": TMD,
                    "ACCI": ACCI
                }
            }
        }
    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }

####################################
### Calculate production profile ###
####################################

@tool
def calculate_production(
    userID: Annotated[str, InjectedState("userID")],
    table_name: Annotated[Optional[str], "The table containing the input data"]=None,
    oil_qi: Annotated[Optional[float], "Oil initial production rate in bbl/d"] = 0.0,
    oil_di: Annotated[Optional[float], "Oil initial decline rate"] = 0.0,
    oil_b: Annotated[Optional[float], "Oil hyperbolic b factor (1 for exponential decline)"] = 0.0,
    gas_qi: Annotated[Optional[float], "Gas initial production rate in mcf/d"] = 0.0,
    gas_di: Annotated[Optional[float], "Gas initial decline rate"] = 0.0,
    gas_b: Annotated[Optional[float], "Gas hyperbolic b factor (1 for exponential decline)"] = 0.0,
    propane_qi: Annotated[Optional[float], "Propane initial production rate in bbl/d"] = 0.0,
    propane_di: Annotated[Optional[float], "Propane initial decline rate"] = 0.0,
    propane_b: Annotated[Optional[float], "Propane hyperbolic b factor (1 for exponential decline)"] = 0.0,
    butane_qi: Annotated[Optional[float], "Butane initial production rate in bbl/d"] = 0.0,
    butane_di: Annotated[Optional[float], "Butane initial decline rate"] = 0.0,
    butane_b: Annotated[Optional[float], "Butane hyperbolic b factor (1 for exponential decline)"] = 0.0,
    forecast_periods: Annotated[Optional[int], "Number of production periods to calculate (months)"] = 120
) -> Dict:
    """
    Calculate the production profile using Arps decline curve parameters.

    This function generates monthly production profiles and calculates EUR (Estimated Ultimate Recovery). It can
    be used for oil, gas, propane (C3), butane (C4), or total BOE (Barrels of Oil Equivalent) calculation.

    ### Returns:
    A dictionary containing:
    - Monthly production profiles for oil, gas, propane, butane, and total BOE.
    - EUR (Estimated Ultimate Recovery) values for each product.
    """
    logger = get_logger(userID )
    logger.info(f"Calculate production profile...<br>")
    print("Calculate production profile...")
    try:
        # Create monthly time array
        time = np.arange(forecast_periods)
      
        # Calculate oil production using hyperbolic decline
        if oil_qi >0:
            if oil_b != 1:
                oil_rate = oil_qi/ (1 + oil_b * oil_di * time) ** (1 / oil_b)
            else:
                oil_rate = oil_qi * np.exp(-oil_di * time)
        
            # Convert daily rates to monthly volumes
            oil_production = oil_rate * 30.4  # Average days per month
        else:
            oil_production = np.zeros(forecast_periods)

        # Calculate gas production using hyperbolic decline
        if gas_qi >0:
            if gas_b != 1:
                gas_rate = gas_qi / (1 + gas_b * gas_di * time) ** (1 / gas_b)
            else:
                gas_rate = gas_qi * np.exp(-gas_di * time)
        
            # Convert daily rates to monthly volumes
            gas_production = gas_rate * 30.4
        else:
            gas_production = np.zeros(forecast_periods)

        # Calculate propane production using hyperbolic decline
        if propane_qi >0:
            if propane_b != 1:
                propane_rate = propane_qi / (1 + propane_b * propane_di * time) ** (1 / propane_b)
            else:
                propane_rate = propane_qi * np.exp(-propane_di * time)
        
            # Convert daily rates to monthly volumes
            propane_production = propane_rate * 30.4
        else:
            propane_production = np.zeros(forecast_periods)

        # Calculate butane production using hyperbolic decline
        if butane_qi >0:
            if butane_b != 1:
                butane_rate = butane_qi / (1 + butane_b * butane_di * time) ** (1 / butane_b)
            else:
                butane_rate = butane_qi * np.exp(-butane_di * time)
        
            # Convert daily rates to monthly volumes
            butane_production = butane_rate * 30.4
        else:
            butane_production = np.zeros(forecast_periods)
        
        boe = (oil_production + gas_production / 6 + propane_production + butane_production)

        # Create production dictionary
        production_data = {
            'prod_month': time.tolist(),
            'oil_production': oil_production.tolist(),
            'gas_production': gas_production.tolist(),
            'propane_production': propane_production.tolist(),
            'butane_production': butane_production.tolist(),
            'boe': boe.tolist()
        }
        
        # Store the production data to the database
        if table_name is None:
            table_name = 'economic_calculation_output_' + uuid.uuid4().hex[:4]
        db_name = os.path.join("database", userID + ".db")
        df = pd.DataFrame(production_data)
        conn = sqlite3.connect(db_name)
        
        # Generate description dynamically with all column names
        all_columns = "\n".join([f"- '{col}'" for col in df.columns])
        description = (
            f"The table includes the following columns:\n{all_columns}\n\n"
            "This table has been updated and saved for further analysis."
        )

        # Store updated data to the database
        store_data_sqlite3(
            filename=userID,
            table="running",
            data=f"Saved table: {table_name}",
            type="processed",
            description=description
        )
        df.to_sql(table_name, conn, if_exists='replace', index=False)

        # Generate message
        new_columns = [
            'prod_month', 'oil_production', 'gas_production','propane_production',
            'butane_production', 'boe'
        ]
        message = (
            f"Production calculation successful. The calculated production data are stored in the database table: '{table_name}'.\n"
            f"The following columns have been added:\n"
            f"{', '.join(new_columns)}."
        )

        output_results =  {
            "message": message,
            "metadata": {
                "eur": sum(boe)
            }
        }
    
        return json.dumps(output_results)
    except Exception as e:
        return {"status": "error", "message": str(e)}
    

###############################
### Calculate royalty rates ###
###############################

@tool
def calculate_combined_royalty_rate(
    userID: Annotated[str, InjectedState("userID")],
    c_star: float,
    table_name: Annotated[Optional[str], "The table containing the production data from which royalty will be calculated"]=None,
    oil_production_col: Annotated[Optional[str], "The column name for oil production in the table"] = None,
    gas_production_col: Annotated[Optional[str], "The column name for gas production in the table"] = None,
    propane_production_col: Annotated[Optional[str], "The column name for propane production in the table"] = None,
    butane_production_col: Annotated[Optional[str], "The column name for butane production in the table"] = None,
    cumulative_total_revenue_col: Annotated[Optional[str], "The column name for cumulative total revenue in the table"] = None,
    oil_production: Annotated[Optional[List[float]],"List of monthly oil production volume (bbl/month)"] = [0],
    gas_production: Annotated[Optional[List[float]],"List of monthly gas production volume (Mcf/month)"] = [0],
    propane_production:Annotated[Optional[List[float]],"List of monthly propane production volume (bbl/month)"] = [0],
    butane_production:Annotated[Optional[List[float]],"List of monthly propane production volume (bbl/month)"] = [0],
    cumulative_total_revenue: Annotated[Optional[List[float]],"List of cumulative revenue at each timestamp, defaults to calculated values"] = None,
    oil_price: Annotated[Optional[List[float]], "List of oil prices per barrel ($/bbl)"] = [0],
    gas_price: Annotated[Optional[List[float]], "List of gas prices per Mcf ($/Mcf)"] = [0],
    propane_price: Annotated[Optional[List[float]], "List of propane prices per barrel ($/bbl)"] = [0],
    butane_price: Annotated[Optional[List[float]], "List of butane prices per barrel ($/bbl)"] = [0],
) -> Dict:
    """
    Calculate royalty rates for oil, gas, propane, and butane.

    You can use table_name and relevant col name to retrieve the saved data for input
    Or you can directly input data from user questions. Please ensure to input all the avialable information you have (table or manually input)
    
    Returns:
    A dictionary with a success message.
    """
    logger = get_logger(userID )
    logger.info(f"Calculate royalty rates...<br>")
    print("Calculate royalty rates...")
    try:
        # Load data
        db_name = os.path.join("database", userID + ".db")
        if table_name is not None:
            df = load_data_to_dataframe(db_name, table_name)
        else:
            df = pd.DataFrame()

        # def broadcast_if_needed(arr, length):
        #     # If we have a single-element array, replicate it
        #     if len(arr) == 1:
        #         arr = np.full(length, arr[0])
        #     elif len(arr) > length:
        #         # If provided arrays are longer than needed, truncate them
        #         arr = arr[:length]
        #     return arr
        print("=======================hi++++++++++++++++++++++++++")

        # Extract and ensure the same length
        # n_periods = len(df)
        if table_name is not None:
            if oil_production_col is not None and oil_production_col in df.columns:
                oil_production = df[oil_production_col].values.tolist()
            
            if gas_production_col is not None and gas_production_col in df.columns:
                gas_production = df[gas_production_col].values.tolist()

            if propane_production_col is not None and propane_production_col in df.columns:
                propane_production = df[propane_production_col].values.tolist()

            if butane_production_col is not None and butane_production_col in df.columns:
                butane_production = df[butane_production_col].values.tolist()

            if cumulative_total_revenue_col is not None and cumulative_total_revenue_col in df.columns:
                cumulative_total_revenue = df[cumulative_total_revenue_col].values.tolist()

        # oil_production = broadcast_if_needed(extract_or_default(df, oil_production_col), n_periods)
        # gas_production = broadcast_if_needed(extract_or_default(df, gas_production_col), n_periods)
        # propane_production = broadcast_if_needed(extract_or_default(df, propane_production_col), n_periods)
        # butane_production = broadcast_if_needed(extract_or_default(df, butane_production_col), n_periods)
        # cumulative_total_revenue = broadcast_if_needed(extract_or_default(df, cumulative_total_revenue_col), n_periods)

        # # Broadcast price arrays to ensure they also match n_periods
        # oil_price = broadcast_if_needed(oil_price, n_periods)
        # gas_price = broadcast_if_needed(gas_price, n_periods)
        # propane_price = broadcast_if_needed(propane_price, n_periods)
        # butane_price = broadcast_if_needed(butane_price, n_periods)

        # Validate input lengths
        print(len(oil_production))
        print(len(gas_production))
        print(len(propane_production))
        print(len(butane_production))
        print(len(oil_price))
        print(len(gas_price))
        print(len(propane_price))
        print(len(butane_price))

        input_len = sorted([len(oil_production), len(gas_production), len(propane_production), len(butane_production), len(oil_price), len(gas_price), len(propane_price), len(butane_price)])
        print(input_len)
        input_len = [l for l in input_len if l > 1]
        print(len(input_len))
        
        if len(input_len) > 0:
            max_len = input_len[-1]
            min_len = input_len[0]   
        else:
            max_len = 1
            min_len = 1

        if min_len != max_len and min_len > 1:
            # Truncate to shortest length
            oil_price = oil_price[:min_len]
            oil_production = oil_production[:min_len]
            gas_price = gas_price[:min_len]
            gas_production = gas_production[:min_len]
            propane_price = propane_price[:min_len]
            propane_production = propane_production[:min_len]
            butane_price = butane_price[:min_len]
            butane_production = butane_production[:min_len]

        # Convert units to metric
        if max(oil_price)>0:
            n_oil_price = [p * 6.29 for p in oil_price]  # Convert $/bbl to $/m3
            n_oil_production = [q / 6.29 for q in oil_production]  # Convert bbl to m3

        if max(gas_price)>0:
            n_gas_price = [p * 0.934 for p in gas_price]  # Convert $/Mcf to $/GJ
            n_gas_production = [q / 35.31 for q in gas_production]  # Convert Mcf to m3
        
        if max(propane_price)>0:
            n_propane_price = [p * 6.29 for p in propane_price] # Convert $/bbl to $/m3
            n_propane_production = [q / 6.29 for q in propane_production] # Convert bbl to m3

        if max(butane_price)>0:
            n_butane_price = [p * 6.29 for p in butane_price] # Convert $/bbl to $/m3
            n_butane_production = [q / 6.29 for q in butane_production] # Convert bbl to m3

        def calculate_oil_price_royalty_rate(oil_pp: float) -> float:
            """Calculate royalty rate based on oil price per m3"""
            if oil_pp <= 251.7:
                return 0.10
            elif 251.7 < oil_pp <= 409.02:
                rate = ((oil_pp - 251.7) * 0.00071 + 0.1)
                return min(rate, 0.40)
            elif 409.02 < oil_pp <= 723.64:
                rate = ((oil_pp - 409.02) * 0.00039 + 0.2117)
                return min(rate, 0.40)
            else:  
                rate = ((oil_pp - 723.64) * 0.00020 + 0.33440)
                return min(rate, 0.40)

        def calculate_gas_price_royalty_rate(gas_pp: float) -> float:
            """Calculate royalty rate based on gas price per GJ"""
            if gas_pp <= 2.4:
                return 0.10
            elif 2.4 < gas_pp <= 3.00:
                rate = ((gas_pp - 2.4) * 0.06000 + 0.05000)
                return min(rate, 0.36)
            elif 3.00 < gas_pp <= 6.75:
                rate = ((gas_pp - 3.00) * 0.04250 + 0.08600)
                return min(rate, 0.36)
            else:  # PP > 6.75
                rate = ((gas_pp - 6.75) * 0.02250 + 0.24538)
                return min(rate, 0.36)
        
        def calculate_propane_price_royalty_rate(propane_pp: float) -> float:
            """Calculate royalty rate based on propane price per m3"""
            if propane_pp <= 88.10:
                return 0.10
            elif 88.10 < propane_pp <= 143.16:
                rate = ((propane_pp - 88.10) * 0.00202 + 0.1)
                return min(rate, 0.36)
            elif 143.16 < propane_pp <= 253.28:
                rate = ((propane_pp - 143.16) * 0.00111 + 0.21122)
                return min(rate, 0.36)
            else:  
                rate = ((propane_pp - 253.28) * 0.00059 + 0.33347)
                return min(rate, 0.36)
        
        def calculate_butane_price_royalty_rate(butane_pp: float) -> float:
            """Calculate royalty rate based on butane price per m3"""
            if butane_pp <= 176.19:
                return 0.10
            elif 176.19 < butane_pp <= 286.31:
                rate = ((butane_pp - 176.19) * 0.00101 + 0.1)
                return min(rate, 0.36)
            elif 286.31 < butane_pp <= 506.55:
                rate = ((butane_pp - 286.31) * 0.00055 + 0.21122)
                return min(rate, 0.36)
            else:  
                rate = ((butane_pp - 506.55) * 0.00031 + 0.33235)
                return min(rate, 0.36)

        def calculate_oil_quantity_adjustment(oil_q: float) -> float:
            """Calculate quantity adjustment for oil volumes"""
            if oil_q >= 194.0:
                return 0.0
            else:
                rq = ((oil_q - 194.0) * 0.001350)
                return max(rq, -1)  # Ensure rq doesn't go below -100%

        def calculate_gas_quantity_adjustment(gas_q: float) -> float:
            """Calculate quantity adjustment for gas volumes"""
            if gas_q >= 345.5:
                return 0.0
            else:
                rq = ((gas_q - 345.5) * 0.0004937)
                return max(rq, -1)  # Ensure rq doesn't go below -100%

        def calculate_propane_quantity_adjustment(propane_q: float) -> float:
            """Calculate quantity adjustment for propane volumes"""
            if propane_q >= 194.0:
                return 0.0
            else:
                rq = ((propane_q - 194.0) * 0.001350)
                return max(rq, -1)
        
        def calculate_butane_quantity_adjustment(butane_q: float) -> float:
            """Calculate quantity adjustment for butane volumes"""
            if butane_q >= 194.0:
                return 0.0
            else:
                rq = ((butane_q - 194.0) * 0.001350)
                return max(rq, -1)
            
        # Initialize arrays for results
        print(oil_price)
        print(len(oil_price))
        n_periods = len(oil_price)
        
        # Oil calculations
        oil_price_rates = np.zeros(n_periods)
        oil_quantity_adjustments = np.zeros(n_periods)
        oil_combined_rates = np.zeros(n_periods)
        oil_final_rates = np.zeros(n_periods)
        oil_price_tiers = []
        
        # Gas calculations
        gas_price_rates = np.zeros(n_periods)
        gas_quantity_adjustments = np.zeros(n_periods)
        gas_combined_rates = np.zeros(n_periods)
        gas_final_rates = np.zeros(n_periods)
        gas_price_tiers = []

        # Propane calculations
        propane_price_rates = np.zeros(n_periods)
        propane_quantity_adjustments = np.zeros(n_periods)
        propane_combined_rates = np.zeros(n_periods)
        propane_final_rates = np.zeros(n_periods)
        propane_price_tiers = []

        # Butane calculations
        butane_price_rates = np.zeros(n_periods)
        butane_quantity_adjustments = np.zeros(n_periods)
        butane_combined_rates = np.zeros(n_periods)
        butane_final_rates = np.zeros(n_periods)
        butane_price_tiers = []
        
        # Calculate cumulative revenue if not provided
        if cumulative_total_revenue is None:
            monthly_total_revenue = (np.array(oil_price) * np.array(oil_production)) + (np.array(gas_price) * np.array(gas_production)) +\
                                    (np.array(propane_price) * np.array(propane_production)) + (np.array(butane_price) * np.array(butane_production))
            cumulative_total_revenue = np.cumsum(monthly_total_revenue).tolist()

        # Calculate rates for each timestamp
        for i in range(n_periods):
            # Calculate oil components
            if max(oil_production) > 0:
                oil_price_rates[i] = calculate_oil_price_royalty_rate(n_oil_price[i])
                oil_quantity_adjustments[i] = calculate_oil_quantity_adjustment(n_oil_production[i])
                oil_combined_rates[i] = oil_price_rates[i] + oil_quantity_adjustments[i]
                
            # Calculate gas components
            if max(gas_production) > 0:
                gas_price_rates[i] = calculate_gas_price_royalty_rate(n_gas_price[i])
                gas_quantity_adjustments[i] = calculate_gas_quantity_adjustment(n_gas_production[i])
                gas_combined_rates[i] = gas_price_rates[i] + gas_quantity_adjustments[i]
                
            # Calculate propane components
            if max(propane_production) > 0:
                propane_price_rates[i] = calculate_propane_price_royalty_rate(n_propane_price[i])
                propane_quantity_adjustments[i] = calculate_propane_quantity_adjustment(n_propane_production[i])
                propane_combined_rates[i] = propane_price_rates[i] + propane_quantity_adjustments[i]

            # Calculate butane components
            if max(butane_production) > 0:
                butane_price_rates[i] = calculate_butane_price_royalty_rate(n_butane_price[i])
                butane_quantity_adjustments[i] = calculate_butane_quantity_adjustment(n_butane_production[i])
                butane_combined_rates[i] = butane_price_rates[i] + butane_quantity_adjustments[i]

            # Determine payout phase
            is_post_payout = cumulative_total_revenue[i] >= c_star
            oil_final_rates[i] = max(oil_combined_rates[i], 0.05) if is_post_payout else 0.05
            gas_final_rates[i] = max(gas_combined_rates[i], 0.05) if is_post_payout else 0.05
            propane_final_rates[i] = max(propane_combined_rates[i], 0.05) if is_post_payout else 0.05
            butane_final_rates[i] = max(butane_combined_rates[i], 0.05) if is_post_payout else 0.05
            
            # Determine oil price tier
            if len(oil_price) > 1:
                oil_price_tiers.append(
                    "1" if oil_price[i] <= 40.02 else
                    "2" if oil_price[i] <= 65.03 else
                    "3" if oil_price[i] <= 115.05 else
                    "4"
                )
            
            # Determine gas price tier
            if len(gas_price) > 1:
                gas_price_tiers.append(
                    "1" if gas_price[i] <= 2.57 else
                    "2" if gas_price[i] <= 3.21 else
                    "3" if gas_price[i] <= 7.23 else
                    "4"
                )

            # Determine propane price tier
            if len(propane_price) > 1:
                propane_price_tiers.append(
                    "1" if propane_price[i] <= 14.00 else
                    "2" if propane_price[i] <= 22.76 else
                    "3" if propane_price[i] <= 40.27 else
                    "4"
                )

            # Determine butane price tier
            if len(butane_price) > 1:
                butane_price_tiers.append(
                    "1" if butane_price[i] <= 28.01 else
                    "2" if butane_price[i] <= 45.52 else
                    "3" if butane_price[i] <= 80.53 else
                    "4"
                )
        print(len(oil_price))
        print(len(gas_price))
        print(len(propane_price))
        print(len(butane_price))

        # Store the production data to the database
        df['oil_final_royalty_rates'] = oil_final_rates
        df['oil_price_based_royalty_rates'] = oil_price_rates
        df['oil_quantity_adjustments_royalty_rates'] = oil_quantity_adjustments
        df['oil_combined_ratesd_royalty_rates'] = oil_combined_rates

        print(oil_combined_rates)

        df['gas_final_royalty_rates'] = gas_final_rates
        df['gas_price_based_royalty_rates'] = gas_price_rates
        df['gas_quantity_adjustments_royalty_rates'] = gas_quantity_adjustments
        df['gas_combined_ratesd_royalty_rates'] = gas_combined_rates

        df['propane_final_royalty_rates'] = propane_final_rates
        df['propane_price_based_royalty_rates'] = propane_price_rates
        df['propane_quantity_adjustments_royalty_rates'] = propane_quantity_adjustments
        df['propane_combined_ratesd_royalty_rates'] = propane_combined_rates

        df['butane_final_royalty_rates'] = butane_final_rates
        df['butane_price_based_royalty_rates'] = butane_price_rates
        df['butane_quantity_adjustments_royalty_rates'] = butane_quantity_adjustments
        df['butane_combined_ratesd_royalty_rates'] = butane_combined_rates

        df['payout phases'] = ["post-payout" if rev >= c_star else "pre-payout" for rev in cumulative_total_revenue]

        # Generate description dynamically with all column names
        all_columns = "\n".join([f"- '{col}'" for col in df.columns])
        description = (
            f"The table includes the following columns:\n{all_columns}\n\n"
            "This table has been updated and saved for further analysis."
        )

        # Store updated DataFrame back to the database
        print(table_name)
        if table_name is None:
            table_name = 'economic_calculation_output_' + uuid.uuid4().hex[:4]

        print(table_name)
        conn = sqlite3.connect(db_name)
        df.to_sql(table_name, conn, if_exists='replace', index=False)

        store_data_sqlite3(
            filename=userID,
            table="running",
            data=f"Saved table: {table_name}",
            type="processed",
            description=description
        )
        
        new_columns = [
            # Oil
            'oil_final_royalty_rates',
            'oil_price_based_royalty_rates',
            'oil_quantity_adjustments_royalty_rates',
            'oil_combined_ratesd_royalty_rates',
            # Gas
            'gas_final_royalty_rates',
            'gas_price_based_royalty_rates',
            'gas_quantity_adjustments_royalty_rates',
            'gas_combined_ratesd_royalty_rates',
            # Propane
            'propane_final_royalty_rates',
            'propane_price_based_royalty_rates',
            'propane_quantity_adjustments_royalty_rates',
            'propane_combined_ratesd_royalty_rates',
            # Butane
            'butane_final_royalty_rates',
            'butane_price_based_royalty_rates',
            'butane_quantity_adjustments_royalty_rates',
            'butane_combined_ratesd_royalty_rates',
            # Additional
            'payout phases'
        ]

        message = (
            f"Royalty rate calculation successful. The updated results have been stored in the database table '{table_name}'.\n"
            f"The following columns have been added:\n"
            f"{', '.join(new_columns)}."
        )
        return {
            "message": message,
            "metadata":{
                "oil_price_tiers": oil_price_tiers,
                "gas_price_tiers": gas_price_tiers,
                "propane_price_tiers": propane_price_tiers,
                "butane_price_tiers": butane_price_tiers
            }
        }
    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }

################################################
### Calculate monthly and cumulative revenue ###
################################################

@tool
def calculate_revenue(
        userID: Annotated[str, InjectedState("userID")],
        table_name: Annotated[Optional[str], "The table containing the production data from which revenue will be calculated"]=None,
        oil_production_col: Annotated[Optional[str], "The column name for oil production in the table"]=None,
        gas_production_col: Annotated[Optional[str], "The column name for gas production in the table"]=None,
        propane_production_col: Annotated[Optional[str], "The column name for propane production in the table"]=None,
        butane_production_col: Annotated[Optional[str], "The column name for butane production in the table"]=None,
        oil_production: Annotated[Optional[List[float]],"List of monthly oil production volume (bbl/month)"] = [0],
        gas_production: Annotated[Optional[List[float]],"List of monthly gas production volume (Mcf/month)"] = [0],
        propane_production:Annotated[Optional[List[float]],"List of monthly propane production volume (bbl/month)"] = [0],
        butane_production:Annotated[Optional[List[float]],"List of monthly propane production volume (bbl/month)"] = [0],
        oil_price: Annotated[Optional[List[float]], "List of oil prices per barrel ($/bbl)"] = [0],
        gas_price: Annotated[Optional[List[float]], "List of gas prices per Mcf ($/Mcf)"] = [0],
        propane_price: Annotated[Optional[List[float]], "List of propane prices per barrel ($/bbl)"] = [0],
        butane_price: Annotated[Optional[List[float]], "List of butane prices per barrel ($/bbl)"] = [0],
    ) -> Dict:
    """
    Calculate monthly and cumulative revenue streams from production profiles based on production data retrieved 
    from a specified database table. 

    You can use table_name and relevant col name to retrieve the saved data for input
    Or you can directly input data from user questions. Please ensure to input all the avialable information you have (table or manually input)

    Returns:
    A dictionary containing:
    - 'monthly_revenues': A breakdown of revenue for each production month.
    - 'cumulative_revenue': The total revenue accumulated over time.
    """
    logger = get_logger(userID )
    logger.info(f"Calculate monthly and cumulative revenue...<br>")
    print("Calculate monthly and cumulative revenue....")
    try:
        # # Load data
        # db_name = os.path.join("database", userID + ".db")
        # df = load_data_to_dataframe(db_name, table_name)

        # # Initialize production arrays
        # oil_production = df[oil_production_col].values if oil_production_col in df.columns else np.zeros(len(df))
        # gas_production = df[gas_production_col].values if gas_production_col in df.columns else np.zeros(len(df))
        # propane_production = df[propane_production_col].values if propane_production_col in df.columns else np.zeros(len(df))
        # butane_production = df[butane_production_col].values if butane_production_col in df.columns else np.zeros(len(df))

         # Load data
        db_name = os.path.join("database", userID + ".db")
        if table_name is not None:
            df = load_data_to_dataframe(db_name, table_name)
        else:
            df = pd.DataFrame()

        # Extract and ensure the same length
        # n_periods = len(df)
        if table_name is not None:
            if oil_production_col is not None and oil_production_col in df.columns:
                oil_production = df[oil_production_col].values.tolist()
            
            if gas_production_col is not None and gas_production_col in df.columns:
                gas_production = df[gas_production_col].values.tolist()

            if propane_production_col is not None and propane_production_col in df.columns:
                propane_production = df[propane_production_col].values.tolist()

            if butane_production_col is not None and butane_production_col in df.columns:
                butane_production = df[butane_production_col].values.tolist()

         # Validate input lengths
        input_len = sorted([len(oil_production), len(gas_production), len(propane_production), len(butane_production), len(oil_price), len(gas_price), len(propane_price), len(butane_price)])
        input_len = [l for l in input_len if l > 1]
        if len(input_len) > 0:
            max_len = input_len[-1]
            min_len = input_len[0]   
        else:
            max_len = 1
            min_len = 1 

        if min_len != max_len and min_len > 1:
            # Truncate to shortest length
            oil_price = oil_price[:min_len]
            oil_production = oil_production[:min_len]
            gas_price = gas_price[:min_len]
            gas_production = gas_production[:min_len]
            propane_price = propane_price[:min_len]
            propane_production = propane_production[:min_len]
            butane_price = butane_price[:min_len]
            butane_production = butane_production[:min_len]

        # Calculate monthly revenues
        monthly_oil_revenue = oil_production * oil_price
        monthly_gas_revenue = gas_production * gas_price
        monthly_propane_revenue = propane_production * propane_price
        monthly_butane_revenue = butane_production * butane_price
        monthly_total_revenue = (
            monthly_oil_revenue + monthly_gas_revenue + monthly_propane_revenue + monthly_butane_revenue
        )
        
        # Calculate cumulative revenues
        cumulative_oil_revenue = np.cumsum(monthly_oil_revenue)
        cumulative_gas_revenue = np.cumsum(monthly_gas_revenue)
        cumulative_propane_revenue = np.cumsum(monthly_propane_revenue)
        cumulative_butane_revenue = np.cumsum(monthly_butane_revenue)
        cumulative_total_revenue = np.cumsum(monthly_total_revenue)

        # Add calculated columns to the dataframe
        df['monthly_oil_revenue'] = monthly_oil_revenue
        df['monthly_gas_revenue'] = monthly_gas_revenue
        df['monthly_propane_revenue'] = monthly_propane_revenue
        df['monthly_butane_revenue'] = monthly_butane_revenue
        df['monthly_total_revenue'] = monthly_total_revenue
        df['cumulative_oil_revenue'] = cumulative_oil_revenue
        df['cumulative_gas_revenue'] = cumulative_gas_revenue
        df['cumulative_propane_revenue'] = cumulative_propane_revenue
        df['cumulative_butane_revenue'] = cumulative_butane_revenue
        df['cumulative_total_revenue'] = cumulative_total_revenue

        # Generate description dynamically with all column names
        all_columns = "\n".join([f"- '{col}'" for col in df.columns])
        description = (
            f"The table includes the following columns:\n{all_columns}\n\n"
            "This table has been updated and saved for further analysis."
        )
        
        # Store updated data to the database
        if table_name is None:
            table_name = 'economic_calculation_output_' + uuid.uuid4().hex[:4]

        store_data_sqlite3(
            filename=userID,
            table="running",
            data=f"Saved table: {table_name}",
            type="processed",
            description=description
        )

        conn = sqlite3.connect(db_name)
        df.to_sql(table_name, conn, if_exists='replace', index=False)

        # Generate message
        new_columns = ['monthly_oil_revenue', 'monthly_gas_revenue', 'monthly_propane_revenue', 'monthly_butane_revenue',
            'monthly_total_revenue', 'cumulative_oil_revenue', 'cumulative_gas_revenue',
            'cumulative_propane_revenue', 'cumulative_butane_revenue', 'cumulative_total_revenue']
        message = (
            f"Revenue calculation successful. The updated results have been stored in the database table '{table_name}'.\n"
            f"The following columns have been added:\n"
            f"{', '.join(new_columns)}."
        )

        return {
            "status": "success",
            "message": message,
            "metadata": {
                "total_oil_revenue": round(float(np.sum(monthly_oil_revenue)), 2),
                "total_gas_revenue": round(float(np.sum(monthly_gas_revenue)), 2),
                "total_propane_revenue": round(float(np.sum(monthly_propane_revenue)), 2),
                "total_butane_revenue": round(float(np.sum(monthly_butane_revenue)), 2),
                "total_revenue": round(float(np.sum(monthly_total_revenue)), 2)
            }
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


#########################################
### Calculate monthly cost components ###
#########################################

@tool
def calculate_monthly_costs(
        userID: Annotated[str, InjectedState("userID")],
        capex: Annotated[float, "Initial capital expenditure"],
        transportation: Annotated[float, "Transportation cost per BOE"],
        table_name: Annotated[Optional[str], "The correct table containing the boe value"] = None,
        boe_col: Annotated[Optional[str], "The column name for BOE data in the table"]=None,
        boe: Annotated[Optional[List[float]], "List of monthly production in BOE"] = [0],
        fixed_opex: Annotated[Optional[float], "Fixed operating cost per month"] = None,
        variable_opex: Annotated[Optional[float], "Variable operating cost per BOE"] = None
    ) -> Dict:
    """
    Calculate monthly cost components based on BOE data retrieved from the table.

    You can use table_name and relevant col name to retrieve the saved data for input
    Or you can directly input data from user questions. Please ensure to input all the avialable information you have (table or manually input)

    ### Returns:
    A dictionary containing:
    - Monthly cost components.
    - Metadata with total costs.
    """
    logger = get_logger(userID )
    logger.info(f"Calculate monthly cost components...<br>")
    print("Calculate monthly cost components....")
    try:
        # Load data
        db_name = os.path.join("database", userID + ".db")
        if table_name is not None:
            df = load_data_to_dataframe(db_name, table_name)
        else:
            df = pd.DataFrame()

        # Extract and ensure the same length
        # n_periods = len(df)
        if table_name is not None:
            if boe_col is not None and boe_col in df.columns:
                boe = df[boe_col].values.tolist()
        
        periods = len(boe)
                
        # Fixed costs array
        if fixed_opex is None:
            eur = sum(boe)
            fixed_opex = eur * 15 * 0.25/120
            fixed_opex_costs = [fixed_opex] * periods
        else:
            fixed_opex_costs = [fixed_opex] * periods

        # Calculate variable costs based on total BOE
        if variable_opex is None:
            variable_opex = eur*15*0.75/eur
            variable_opex_costs = [b * variable_opex for b in boe]
        else:
            variable_opex_costs = [b * variable_opex for b in boe]
        
        # Total opex
        total_opex = [f + v for f, v in zip(fixed_opex_costs, variable_opex_costs)]
        
        # Transportation costs
        transportation_costs = [b * transportation for b in boe]
        
        # Capex (assumed to be incurred at start)
        capex_costs = [capex] + [0] * (periods - 1)
        
        # Append calculated columns to the DataFrame
        df['monthly_fixed_opex'] = fixed_opex_costs
        df['monthly_variable_opex'] = variable_opex_costs
        df['monthly_total_opex'] = total_opex
        df['monthly_transportation'] = transportation_costs
        df['monthly_capex'] = capex_costs

        # Generate description dynamically with all column names
        all_columns = "\n".join([f"- '{col}'" for col in df.columns])
        description = (
            f"The table includes the following columns:\n{all_columns}\n\n"
            "This table has been updated and saved for further analysis."
        )

        # Store updated DataFrame back to the database
        if table_name is not None:
            table_name = 'economic_calculation_output_' + uuid.uuid4().hex[:4]
        
        conn = sqlite3.connect(db_name)
        df.to_sql(table_name, conn, if_exists='replace', index=False)

        store_data_sqlite3(
            filename=userID,
            table="running",
            data=f"Saved table: {table_name}",
            type="processed",
            description=description
        )

        # Generate message
        new_columns = [
            'monthly_fixed_opex', 'monthly_variable_opex', 'monthly_total_opex',
            'monthly_transportation', 'monthly_capex'
        ]
        message = (
            f"Monthly cost component calculation successful. The updated results have been stored in the database table '{table_name}'.\n"
            f"The following columns have been added:\n"
            f"{', '.join(new_columns)}."
        )
        
        return {
            "message": message,
            "metadata": {
                "total_capex": sum(capex_costs),
                "total_opex": sum(total_opex),
                "total_transportation": sum(transportation_costs)
            }
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


####################################
### Calculate monthly cash flows ###
####################################

@tool
def calculate_monthly_cashflow(
        userID: Annotated[str, InjectedState("userID")],
        table_name: Annotated[Optional[str], "The correct table containing the BOE value"]=None,
        monthly_total_opex_col: Annotated[Optional[str], "Column name for monthly total operating expenses"]=None,
        monthly_transportation_col: Annotated[Optional[str], "Column name for monthly transportation costs"]=None,
        monthly_capex_col: Annotated[Optional[str], "Column name for monthly capital expenditures"]=None,
        monthly_oil_revenue_col: Annotated[Optional[str], "Column name for monthly oil revenue"] = None,
        monthly_gas_revenue_col: Annotated[Optional[str], "Column name for monthly gas revenue"] = None,
        monthly_propane_revenue_col: Annotated[Optional[str], "Column name for monthly propane revenue"] = None,
        monthly_butane_revenue_col: Annotated[Optional[str], "Column name for monthly butane revenue"] = None,
        oil_royalty_rate_col: Annotated[Optional[str], "Column name for oil royalty rate"] = None,
        gas_royalty_rate_col: Annotated[Optional[str], "Column name for gas royalty rate"] = None,
        propane_royalty_rate_col: Annotated[Optional[str], "Column name for propane royalty rate"] = None,
        butane_royalty_rate_col: Annotated[Optional[str], "Column name for butane royalty rate"] = None,
        monthly_total_opex: Annotated[Optional[List[float]], "List of monthly total operating expenses"] = [0],
        monthly_transportation: Annotated[Optional[List[float]], "List of monthly transportation costs"] = [0],
        monthly_capex: Annotated[Optional[List[float]], "List of monthly capital expenditures"] = [0],
        monthly_oil_revenue: Annotated[Optional[List[float]], "List of monthly oil revenue"] = [0],
        monthly_gas_revenue: Annotated[Optional[List[float]], "List of monthly gas revenue"] = [0],
        monthly_propane_revenue: Annotated[Optional[List[float]], "List of monthly propane revenue"] = [0],
        monthly_butane_revenue: Annotated[Optional[List[float]], "List of monthly butane revenue"] = [0],
        oil_royalty_rate: Annotated[Optional[List[float]], "List of monthly oil royalty rates"] = [0],
        gas_royalty_rate: Annotated[Optional[List[float]], "List of monthly gas royalty rates"] = [0],
        propane_royalty_rate: Annotated[Optional[List[float]], "List of monthly propane royalty rates"] = [0],
        butane_royalty_rate: Annotated[Optional[List[float]], "List of monthly butane royalty rates"] = [0]
    ) -> Dict:
    """
    Calculate monthly cash flows based on data retrieved from the database table.

    You can use table_name and relevant col name to retrieve the saved data for input
    Or you can directly input data from user questions. Please ensure to input all the avialable information you have (table or manually input)

    ### Returns:
    A dictionary containing:
    - Monthly operating cash flows and free cash flows.
    - Total royalties.
    - Metadata with summary statistics.
    """
    logger = get_logger(userID )
    logger.info(f"Calculate monthly cash flows...<br>")
    print("Calculate monthly cash flows....")
    try:
        # Load data
        db_name = os.path.join("database", userID + ".db")
        if table_name is not None:
            df = load_data_to_dataframe(db_name, table_name)
        else:
            df = pd.DataFrame()

        if table_name is not None:
            if monthly_total_opex_col is not None and monthly_total_opex_col in df.columns:
                monthly_total_opex = df[monthly_total_opex_col].values.tolist()

            if monthly_transportation_col is not None and monthly_transportation_col in df.columns:
                monthly_transportation = df[monthly_transportation_col].values.tolist()

            if monthly_capex_col is not None and monthly_capex_col in df.columns:
                monthly_capex = df[monthly_capex_col].values.tolist()

            if monthly_oil_revenue_col is not None and monthly_oil_revenue_col in df.columns:
                monthly_oil_revenue = df[monthly_oil_revenue_col].values.tolist()

            if monthly_gas_revenue_col is not None and monthly_gas_revenue_col in df.columns:
                monthly_gas_revenue = df[monthly_gas_revenue_col].values.tolist()

            if monthly_propane_revenue_col is not None and monthly_propane_revenue_col in df.columns:
                monthly_propane_revenue = df[monthly_propane_revenue_col].values.tolist()

            if monthly_butane_revenue_col is not None and monthly_butane_revenue_col in df.columns:
                monthly_butane_revenue = df[monthly_butane_revenue_col].values.tolist()

            if oil_royalty_rate_col is not None and oil_royalty_rate_col in df.columns:
                oil_royalty_rate = df[oil_royalty_rate_col].values.tolist()
            
            if gas_royalty_rate_col is not None and gas_royalty_rate_col in df.columns:
                gas_royalty_rate = df[gas_royalty_rate_col].values.tolist()
            
            if propane_royalty_rate_col is not None and propane_royalty_rate_col in df.columns:
                propane_royalty_rate = df[propane_royalty_rate_col].values.tolist()

            if butane_royalty_rate_col is not None and butane_royalty_rate_col in df.columns:
                butane_royalty_rate = df[butane_royalty_rate_col].values.tolist

        # Validate input lengths
        input_len = sorted([len(monthly_oil_revenue), len(monthly_gas_revenue), len(monthly_propane_revenue), len(monthly_butane_revenue), len(monthly_total_opex), len(monthly_transportation),
                        len(oil_royalty_rate), len(gas_royalty_rate), len(propane_royalty_rate), len(butane_royalty_rate), len(monthly_capex) if monthly_capex is not None else 0])
        input_len = [l for l in input_len if l > 1]
        max_len = input_len[-1]
        min_len = input_len[0]        

        if min_len != max_len and min_len > 1:
            # Truncate to shortest length
            monthly_oil_revenue = monthly_oil_revenue[:min_len]
            monthly_gas_revenue = monthly_gas_revenue[:min_len]
            monthly_propane_revenue = monthly_propane_revenue[:min_len]
            monthly_butane_revenue = monthly_butane_revenue[:min_len]
            monthly_total_opex = monthly_total_opex[:min_len]
            monthly_transportation = monthly_transportation[:min_len]
            oil_royalty_rate = oil_royalty_rate[:min_len]
            gas_royalty_rate = gas_royalty_rate[:min_len]
            propane_royalty_rate = propane_royalty_rate[:min_len]
            butane_royalty_rate = butane_royalty_rate[:min_len]
            if monthly_capex is not None:
                monthly_capex = monthly_capex[:min_len]

        monthly_total_revenue = np.array(monthly_oil_revenue) + np.array(monthly_gas_revenue) + np.array(monthly_propane_revenue) + np.array(monthly_butane_revenue)
        periods = len(monthly_oil_revenue)
        
        # If capex is not provided, initialize it as a list of zeros
        if monthly_capex is None:
            monthly_capex = [0.0] * periods
        
        # Calculate royalties
        # royalties = [rev * royalty_rate for rev in total_revenue]
        oil_royalties = np.array(monthly_oil_revenue) * np.array(oil_royalty_rate)
        gas_royalties = np.array(monthly_gas_revenue) * np.array(gas_royalty_rate)
        propane_royalties = np.array(monthly_propane_revenue) * np.array(propane_royalty_rate)
        butane_royalties = np.array(monthly_butane_revenue) * np.array(butane_royalty_rate)

        royalties = (oil_royalties + gas_royalties + propane_royalties + butane_royalties).tolist()

        # Calculate monthly cash flows
        free_cash_flows = []
        operating_cash_flows = []  # Before capex
        
        for i in range(periods):
            # Operating cash flow
            operating_cf = (
                monthly_total_revenue[i] -  # Revenue
                monthly_total_opex[i] -     # Opex
                monthly_transportation[i] - # Transportation
                royalties[i]        # Royalties
            )
            operating_cash_flows.append(operating_cf)
            
            # Free cash flow (including capex)
            cf = operating_cf - monthly_capex[i]
            free_cash_flows.append(cf)
        
        # Append calculated columns to the DataFrame
        df['monthly_operating_cash_flows'] = operating_cash_flows
        df['monthly_free_cash_flows'] = free_cash_flows
        df['royalties'] = royalties

        # Generate description dynamically with all column names
        all_columns = "\n".join([f"- '{col}'" for col in df.columns])
        description = (
            f"This table contains production, cost, and cash flow data. "
            f"The table includes the following columns:\n{all_columns}\n\n"
            "This table has been updated and saved for further analysis."
        )

        # Store updated DataFrame back to the database
        if table_name is None:
            table_name = 'economic_calculation_output_' + uuid.uuid4().hex[:4]

        conn = sqlite3.connect(db_name)
        df.to_sql(table_name, conn, if_exists='replace', index=False)

        store_data_sqlite3(
            filename=userID,
            table="running",
            data=f"Saved table: {table_name}",
            type="processed",
            description=description
        )

        # Generate message
        new_columns = [
            'monthly_operating_cash_flows', 'monthly_free_cash_flows', 'royalties'
        ]
        message = (
            f"Cash flow calculation successful. The updated results have been stored in the database table '{table_name}'.\n"
            f"The following columns have been added:\n"
            f"{', '.join(new_columns)}."
        )

        return {
            "message": message,
            "metadata": {
                "total_operating_cf": round(float(sum(operating_cash_flows)),2),
                "total_cash_flow": round(float(sum(free_cash_flows)),2),
                "total_royalties": round(float(sum(royalties)),2)
            }
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}

##################################
### Calculate economic metrics ###
##################################

@tool
def calculate_metrics(
    userID: Annotated[str, InjectedState("userID")],
    table_name: Annotated[Optional[str], "The name of the table containing the cash flow data"]=None,
    monthly_free_cash_flows_col: Annotated[Optional[str], "The column name for the monthly free cash flow"]=None,
    monthly_free_cash_flows: Annotated[Optional[List[float]], "List of monthly free cash flows"]=[0],
    discount_rate: Annotated[Optional[float], "Annual discount rate (default 10%)" ] = 0.1
) -> Dict:
    """
    Calculate economic metrics using monthly cash flows
    
    You can use table_name and relevant col name to retrieve the saved data for input
    Or you can directly input data from user questions. Please ensure to input all the avialable information you have (table or manually input)

    Returns a dictionary with NPV, IRR, payback period, and profitability index
    """
    print("Calculate economic metrics....")
    try:
        # Load data
        db_name = os.path.join("database", userID + ".db")
        if table_name is not None:
            df = load_data_to_dataframe(db_name, table_name)
        else:
            df = pd.DataFrame()

        # Extract and ensure the same length
        # n_periods = len(df)
        if table_name is not None:
            if monthly_free_cash_flows_col is not None and monthly_free_cash_flows_col in df.columns:
                monthly_free_cash_flows = df[monthly_free_cash_flows_col].values.tolist()

        # Convert to numpy array for calculations
        cf_array = np.array(monthly_free_cash_flows)
        
        # Calculate NPV using monthly discount rate
        monthly_rate = discount_rate / 12
        npv = npf.npv(monthly_rate, cf_array)
        
        # Calculate monthly IRR
        monthly_irr = npf.irr(cf_array)
        annual_irr = (1 + monthly_irr) ** 12 - 1 if monthly_irr is not None else None
        
        # Calculate payback period (in months)
        cumulative_cf = np.cumsum(cf_array)
        payback_period = None
        if any(cumulative_cf >= 0):
            payback_period = np.where(cumulative_cf >= 0)[0][0]
        
        # Calculate profitability index
        initial_investment = abs(cf_array[0])
        pi = (npv + initial_investment) / initial_investment if initial_investment != 0 else None

        # Append calculated columns to the DataFrame
        df['cumulative_cash_flow'] = cumulative_cf
        df['npv'] = np.full(len(df), npv)
        df['monthly_irr'] = np.full(len(df), monthly_irr * 100 if monthly_irr is not None else None)
        df['annual_irr'] = np.full(len(df), annual_irr * 100 if annual_irr is not None else None)
        df['payback_period_months'] = np.full(len(df), payback_period if payback_period is not None else None)
        df['profitability_index'] = np.full(len(df), pi)

        # Generate description dynamically with all column names
        all_columns = "\n".join([f"- '{col}'" for col in df.columns])
        description = (
            f"This table contains economic metrics based on the cash flow data. "
            f"The table includes the following columns:\n{all_columns}\n\n"
            "This table has been updated and saved for further analysis."
        )

        # Store updated DataFrame back to the database
        if table_name is None:
            table_name = 'economic_calculation_output_' + uuid.uuid4().hex[:4]

        conn = sqlite3.connect(db_name)
        df.to_sql(table_name, conn, if_exists='replace', index=False)

        # Store processed table metadata
        store_data_sqlite3(
            filename=userID,
            table="running",
            data=f"Saved table: {table_name}",
            type="processed",
            description=description
        )

        # Generate message with new columns
        new_columns = [
            'cumulative_cash_flow', 'npv', 'monthly_irr', 'annual_irr',
            'payback_period_months', 'profitability_index'
        ]
        message = (
            f"Economic metrics calculation successful. The updated results have been stored in the database table '{table_name}'.\n"
            f"The following columns have been added:\n"
            f"{', '.join(new_columns)}."
        )
        
        return {
            "message": message,
            "metadata": {
                "annual_discount_rate": discount_rate,
                "monthly_discount_rate": monthly_rate,
                "analysis_period_months": int(len(cf_array))
            }
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}