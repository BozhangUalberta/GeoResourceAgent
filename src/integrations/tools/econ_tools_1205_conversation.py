from langchain.tools import tool
import numpy as np
import numpy_financial as npf
from typing import Dict,Optional, Union, List
import numpy as np
import numpy_financial as npf
import pandas as pd
import sqlite3

## Load SQL table data to df
def load_data_to_dataframe(db_name, table_name):
    # Load the entire table into a DataFrame
    conn = sqlite3.connect(db_name)
    df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
    conn.close()
    return df

# Extract optional data
def extract_or_default(df, col_name, default=[0]):
    return df[col_name].values if col_name in df.columns else default


@tool
def calculate_well_c_star(
    TVDmax: float,
    TLL: float,
    TPP: float,
    TMD: float,
    TVDavg: Optional[float] = None,
    ACCI: Optional[float] = 1.0
) -> Dict[str, Union[str, float, Dict]]:
    """
    Calculate well c_star (C*) based on drilling and completion parameters
    c_star is a index used to determine oil and gas royality rates, only in Alberta.

    Parameters:
    - TVDmax: Deepest True Vertical Depth (m)
    - TLL: Total Lateral Length (m)
    - TPP: Total Equivalent Proppant Placed (tonnes)
    - TMD: Total Measured Depth (m)
    - TVDavg: Average True Vertical Depth (m), defaults to TVDmax if not provided
    - ACCI: Alberta Capital Cost Index, defaults to 1.0

    Returns a dictionary with calculation results and metadata
    """
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
    
@tool
def calculate_combined_royalty_rate_cov(
    c_star: float,
    oil_price:  Optional[List[float]]=[0],
    oil_production:  Optional[List[float]]=[0],
    gas_price:  Optional[List[float]]=[0], 
    gas_production:  Optional[List[float]]=[0],
    propane_price:  Optional[List[float]]=[0],
    propane_production:  Optional[List[float]]=[0],
    butane_price:  Optional[List[float]]=[0],
    butane_production:  Optional[List[float]]=[0],
    cumulative_total_revenue: Optional[List[float]] = None
) -> Dict[str, Union[str, List[float], Dict]]:
    """
    CONVERSATION SOURCE TOOL
    Calculate both oil and gas royalty rates based on price and quantity parameters for each timestamp

    Parameters:
    - oil_price: List of oil prices per barrel ($/bbl)
    - oil_production: List of monthly oil production volume (bbl/month)
    - gas_price: List of gas prices per Mcf ($/Mcf)
    - gas_production: List of monthly gas production volume (Mcf/month)
    - propane_price: List of propane prices per barrel ($/bbl)
    - propane_production: List of monthly propane production volume (bbl/month)
    - butane_price: List of butane prices per barrel ($/bbl)
    - butane_production: List of monthly butane production volume (bbl/month)
    - c_star: calculated c_star value
    - cumulative_total_revenue: List of cumulative revenue at each timestamp, defaults to calculated values

    Returns a dictionary with calculation results and detailed breakdown for each timestamp for both oil and gas
    """
    try:
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

        return {
            "status": "success",
            "data": {
                "oil": {
                    "royalty_rates": oil_final_rates.tolist(),
                    "formatted_rates": [f"{rate * 100:.2f}%" for rate in oil_final_rates]
                },
                "gas": {
                    "royalty_rates": gas_final_rates.tolist(),
                    "formatted_rates": [f"{rate * 100:.2f}%" for rate in gas_final_rates]
                },
                "propane": {
                    "royalty_rates": propane_final_rates.tolist(),
                    "formatted_rates": [f"{rate * 100:.2f}%" for rate in propane_final_rates]
                },
                "butane": {
                    "royalty_rates": butane_final_rates.tolist(),
                    "formatted_rates": [f"{rate * 100:.2f}%" for rate in butane_final_rates]
                }
            },
            "metadata": {
                "oil": {
                    "price_based_rates": oil_price_rates.tolist(),
                    "quantity_adjustments": oil_quantity_adjustments.tolist(),
                    "combined_rates": oil_combined_rates.tolist(),
                    "inputs": {
                        "prices_per_bbl": oil_price,
                        "quantities_bbl": oil_production,
                    },
                    "price_tiers": {
                        "tiers": oil_price_tiers,
                        "base_rates": [rate * 100 for rate in oil_price_rates],
                        "max_rate": 40.0
                    },
                    "quantity_tiers": {
                        "is_adjusted": [q < 30.84 for q in oil_production],
                        "adjustment_percentages": [adj * 100 for adj in oil_quantity_adjustments],
                        "threshold": 30.84
                    }
                },
                "gas": {
                    "price_based_rates": gas_price_rates.tolist(),
                    "quantity_adjustments": gas_quantity_adjustments.tolist(),
                    "combined_rates": gas_combined_rates.tolist(),
                    "inputs": {
                        "prices_per_mcf": gas_price,
                        "quantities_mcf": gas_production,
                    },
                    "price_tiers": {
                        "tiers": gas_price_tiers,
                        "base_rates": [rate * 100 for rate in gas_price_rates],
                        "max_rate": 36.0
                    },
                    "quantity_tiers": {
                        "is_adjusted": [q < 12199.31 for q in gas_production],
                        "adjustment_percentages": [adj * 100 for adj in gas_quantity_adjustments],
                        "threshold": 12199.31
                    }
                },
                "propane": {
                    "price_based_rates": propane_price_rates.tolist(),
                    "quantity_adjustments": propane_quantity_adjustments.tolist(),
                    "combined_rates": propane_combined_rates.tolist(),
                    "inputs": {
                        "prices_per_bbl": propane_price,
                        "quantities_bbl": propane_production,
                    },
                    "price_tiers": {
                        "tiers": propane_price_tiers,
                        "base_rates": [rate * 100 for rate in propane_price_rates],
                        "max_rate": 36.0
                    },
                    "quantity_tiers": {
                        "is_adjusted": [q < 30.84 for q in propane_production],
                        "adjustment_percentages": [adj * 100 for adj in propane_quantity_adjustments],
                        "threshold": 30.84
                    }
                },
                "butane": {
                    "price_based_rates": butane_price_rates.tolist(),
                    "quantity_adjustments": butane_quantity_adjustments.tolist(),
                    "combined_rates": butane_combined_rates.tolist(),
                    "inputs": {
                        "prices_per_bbl": butane_price,
                        "quantities_bbl": butane_production,
                    },
                    "price_tiers": {
                        "tiers": butane_price_tiers,
                        "base_rates": [rate * 100 for rate in butane_price_rates],
                        "max_rate": 36.0
                    },
                    "quantity_tiers": {
                        "is_adjusted": [q < 30.84 for q in butane_production],
                        "adjustment_percentages": [adj * 100 for adj in butane_quantity_adjustments],
                        "threshold": 30.84
                    }
                },
                "common": {
                    "c_star": c_star,
                    "cumulative_revenue": cumulative_total_revenue,
                    "phases": ["post-payout" if rev >= c_star else "pre-payout" for rev in cumulative_total_revenue]
                }
            }
        }
    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }

@tool
def calculate_production(
    oil_qi: Optional[float]=0.0,
    oil_di: Optional[float]=0.0,
    oil_b: Optional[float]=0.0,
    gas_qi: Optional[float]=0.0,
    gas_di: Optional[float]=0.0,
    gas_b: Optional[float]=0.0,
    propane_qi: Optional[float]=0.0,
    propane_di: Optional[float]=0.0,
    propane_b: Optional[float]=0.0,
    butane_qi: Optional[float]=0.0,
    butane_di: Optional[float]=0.0,
    butane_b: Optional[float]=0.0,
    forecast_periods: Optional[int]=120) -> Dict:
    """
    Calculate production profile using Arps decline curve
    Returns monthly production profiles and EUR 
    Can be used for oil, gas, propane(c3), butane(c4) or total BOE calculation
    Parameters:
    - oil_qi: Oil Initial production rate in bbl/d
    - oil_di: Oil Initial decline rate in
    - oil_b: Oil Hyperbolic b factor (1 for exponential decline)
    - gas_qi: Gas Initial production rate in mcf/d
    - gas_di: Gas Initial decline rate in
    - gas_b: Gas Hyperbolic b factor (1 for exponential decline)
    - propane_qi: Propane Initial production rate in bbl/d
    - propane_di: Propane Initial decline rate in
    - propane_b: Propane Hyperbolic b factor (1 for exponential decline)
    - butane_qi: Butane Initial production rate in bbl/d
    - butane_di: Butane Initial decline rate in
    - butane_b: Butane Hyperbolic b factor (1 for exponential decline)
    - forecast_periods: Number of production periods to calculate (months)

    Returns a dictionary with monthly oil, gas, propane,butane and BOE production profiles
    """
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
        
        return {
            "status": "success",
            "data": production_data,
            "metadata": {
                "eur": sum(boe)
            }
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


@tool
def calculate_revenue(oil_production: Optional[List[float]] = [0],
                      gas_production: Optional[List[float]] = [0],
                      propane_production: Optional[List[float]] = [0],
                      butane_production: Optional[List[float]] = [0],
                      oil_price: Optional[List[float]] = [0],
                      gas_price: Optional[List[float]] = [0],
                      propane_price: Optional[List[float]] = [0],
                      butane_price: Optional[List[float]] = [0]) -> Dict:
    """
    Calculate monthly and cumulative revenue streams from production profiles including cumulative revenues

    Parameters:
    - oil_production: Monthly oil production in bbl
    - gas_production: Monthly gas production in mcf
    - propane_production: Monthly propane production in bbl
    - butane_production: Monthly butane production in bbl
    - oil_price: Price of oil in $/bbl
    - gas_price: Price of gas in $/mcf
    - propane_price: Price of propane in $/bbl
    - butane_price: Price of butane in $/bbl

    Returns:
    Dictionary containing monthly and cumulative revenue streams
    """
    try:
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
        monthly_oil_revenue = (np.array(oil_production) * np.array(oil_price))
        monthly_gas_revenue = (np.array(gas_production) * np.array(gas_price))
        monthly_propane_revenue = (np.array(propane_production) * np.array(propane_price))
        monthly_butane_revenue = (np.array(butane_production) * np.array(butane_price))
        monthly_total_revenue = (monthly_oil_revenue + monthly_gas_revenue + monthly_propane_revenue + monthly_butane_revenue)
        
        # Calculate cumulative revenues
        cumulative_oil_revenue = np.cumsum(monthly_oil_revenue)
        cumulative_gas_revenue = np.cumsum(monthly_gas_revenue)
        cumulative_propane_revenue = np.cumsum(monthly_propane_revenue)
        cumulative_butane_revenue = np.cumsum(monthly_butane_revenue)
        cumulative_total_revenue = np.cumsum(monthly_total_revenue)
        
        return {
            "status": "success",
            "data": {
                "monthly": {
                    "monthly_oil_revenue": monthly_oil_revenue.tolist(),
                    "monthly_gas_revenue": monthly_gas_revenue.tolist(),
                    "monthly_propane_revenue": monthly_propane_revenue.tolist(),
                    "monthly_butane_revenue": monthly_butane_revenue.tolist(),
                    "monthly_total_revenue": monthly_total_revenue.tolist()
                },
                "cumulative": {
                    "cumulative_oil_revenue": cumulative_oil_revenue.tolist(),
                    "cumulative_gas_revenue": cumulative_gas_revenue.tolist(),
                    "cumulative_propane_revenue": cumulative_propane_revenue.tolist(),
                    "cumulative_butane_revenue": cumulative_butane_revenue.tolist(),
                    "cumulative_total_revenue": cumulative_total_revenue.tolist()
                }
            },
            "metadata": {
                "total_oil_revenue": round(float(sum(monthly_oil_revenue)),2),
                "total_gas_revenue": round(float(sum(monthly_gas_revenue)),2),
                "total_propane_revenue": round(float(sum(monthly_propane_revenue)),2),
                "total_butane_revenue": round(float(sum(monthly_butane_revenue)),2),
                "total_revenue": round(float(sum(monthly_total_revenue)),2)
            }
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}

@tool
def calculate_monthly_costs(boe: list[float],
                            capex: float,
                            transportation: float,
                            fixed_opex: Optional[float]=None,
                            variable_opex: Optional[float]=None) -> Dict:
    """
    Calculate monthly total capex, total opex and total transportation costs

    Parameters:
    - boe : Monthly production in BOE
    - capex: Initial capital expenditure
    - fixed_opex: Fixed operating cost $ per month
    - variable_opex: Variable operating cost $ per BOE
    - transportation: Transportation cost $ per BOE

    Returns a dictionary with monthly cost components
    """
    try:
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
        
        cost_data = {
            'monthly_fixed_opex': fixed_opex_costs,
            'monthly_variable_opex': variable_opex_costs,
            'monthly_total_opex': total_opex,
            'monthly_transportation': transportation_costs,
            'monthly_capex': capex_costs
        }
        
        return {
            "status": "success",
            "data": cost_data,
            "metadata": {
                "total_capex": sum(capex_costs),
                "total_opex": sum(total_opex),
                "total_transportation": sum(transportation_costs)
            }
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}

@tool
def calculate_monthly_cashflow(monthly_total_opex: list,
                               monthly_transportation: list, 
                               monthly_oil_revenue: Optional[list[float]]=[0],
                               monthly_gas_revenue: Optional[list[float]]=[0],
                               monthly_propane_revenue: Optional[list[float]]=[0],
                               monthly_butane_revenue: Optional[list[float]]=[0],
                               oil_royalty_rate: Optional[list[float]]=[0],
                               gas_royalty_rate: Optional[list[float]]=[0],
                               propane_royalty_rate: Optional[list[float]]=[0],
                               butane_royalty_rate: Optional[list[float]]=[0],
                               monthly_capex: Optional[List[float]] = None) -> Dict:
    """
    Calculate monthly cash flows

    Parameters:
    - monthly_oil_revenue: Monthly oil revenue
    - monthly_gas_revenue: Monthly gas revenue
    - monthly_propane_revenue: Monthly propane revenue
    - monthly_butane_revenue: Monthly butane revenue
    - total_opex: Monthly total operating expenses
    - transportation: Monthly transportation costs
    - monthly_capex: Monthly capital expenditures
    - oil_royalty_rate: Oil royalty rate as a percentage of oil revenue
    - gas_royalty_rate: Gas royalty rate as a percentage of gas revenue
    - propane_royalty_rate: Propane royalty rate as a percentage of propane revenue
    - butane_royalty_rate: Butane royalty rate as a percentage of butane revenue
    
    Returns a dictionary with monthly cash flows and operating cash flows
    """
    try:
        # Validate input lengths
        input_len = sorted([len(monthly_oil_revenue), len(monthly_gas_revenue), len(monthly_propane_revenue), len(monthly_butane_revenue), len(monthly_total_opex), len(monthly_transportation),
                        len(oil_royalty_rate), len(gas_royalty_rate), len(propane_royalty_rate), len(butane_royalty_rate), len(monthly_capex) if monthly_capex is not None else 0])
        input_len = [l for l in input_len if l > 1]
        if len(input_len) > 0:
            max_len = input_len[-1]
            min_len = input_len[0]   
        else:
            max_len = 1
            min_len = 1     

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
        
        return {
            "status": "success",
            "data": {
                "monthly_operating_cash_flows": operating_cash_flows,
                "monthly_free_cash_flows": free_cash_flows,
                "royalties": royalties
            },
            "metadata": {
                "total_operating_cf": round(float(sum(operating_cash_flows)),2),
                "total_cash_flow": round(float(sum(free_cash_flows)),2),
                "total_royalties": round(float(sum(royalties)),2)
            }
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}

@tool
def calculate_metrics(monthly_free_cash_flows: list, discount_rate: float = 0.10) -> Dict:
    """
    Calculate economic metrics using monthly cash flows

    Parameters:
    - monthly_free_cash_flows: Monthly free cash flows
    - discount_rate: Annual discount rate (default 10%)

    Returns a dictionary with NPV, IRR, payback period, and profitability index
    """
    try:
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
        
        return {
            "status": "success",
            "data": {
                "npv": round(float(npv),2),
                "monthly_irr": monthly_irr * 100 if monthly_irr is not None else None,
                "annual_irr": annual_irr * 100 if annual_irr is not None else None,
                "payback_period_months": int(payback_period),
                "payback_period_years": round(float(payback_period / 12),2) if payback_period is not None else None,
                "profitability_index": round(float(pi),2),
                "cumulative_cash_flow": cumulative_cf.tolist()
            },
            "metadata": {
                "annual_discount_rate": discount_rate,
                "monthly_discount_rate": monthly_rate,
                "analysis_period_months": int(len(cf_array))
            }
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}
    