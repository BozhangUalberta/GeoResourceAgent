from langchain.tools import tool
import numpy as np
import numpy_financial as npf
from typing import Dict

@tool
def calculate_production(initial_rate: float, di: float, b: float, gor: float, periods: int) -> Dict:
    """
    Calculate production profile using Arps decline curve
    Returns monthly production profiles

    Parameters:
    - initial_rate: Initial production rate in bbl/d
    - di: Initial decline rate in
    - b: Hyperbolic b factor (1 for exponential decline)
    - gor: Gas-oil ratio in scf/bbl
    - periods: Number of production periods (months)

    Returns a dictionary with monthly oil, gas, and total BOE production profiles
    """
    try:
        # Create monthly time array
        time = np.arange(periods)
      
        # Calculate oil production using hyperbolic decline
        if b != 1:
            oil_rate = initial_rate / (1 + b * di * time / 12) ** (1 / b)
        else:
            oil_rate = initial_rate * np.exp(-di * time / 12)
        
        # Convert daily rates to monthly volumes
        oil_production = oil_rate * 30.4  # Average days per month
        
        # Calculate gas production using GOR
        gas_production = oil_production * gor / 1000  # Convert scf to mcf
        
        # Create production dictionary
        production_data = {
            'month': time.tolist(),
            'oil_production': oil_production.tolist(),
            'gas_production': gas_production.tolist(),
            'total_boe': (oil_production + gas_production / 6).tolist()  # Convert gas to BOE
        }
        
        return {
            "status": "success",
            "data": production_data,
            "metadata": {
                "eur_oil": sum(oil_production),
                "eur_gas": sum(gas_production),
                "peak_rate": oil_rate[0]
            }
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}

@tool
def calculate_revenue(oil_production: list, gas_production: list, oil_price: float, gas_price: float, scenarios: Dict[str, Dict[str, float]] = None) -> Dict:
    """
    Calculate monthly revenue streams from production profiles

    Parameters:
    - oil_production: Monthly oil production in bbl
    - gas_production: Monthly gas production in mcf
    - oil_price: Price of oil in $/bbl
    - gas_price: Price of gas in $/mcf
    - scenarios: Dictionary of additional price scenarios

    Returns a dictionary with monthly revenue streams for base case and scenarios
    """
    try:
        base_case = {
            'oil_revenue': [p * oil_price for p in oil_production],
            'gas_revenue': [p * gas_price for p in gas_production],
        }
        base_case['total_revenue'] = [o + g for o, g in zip(base_case['oil_revenue'], base_case['gas_revenue'])]
        
        # Handle additional price scenarios if provided
        scenario_results = {}
        if scenarios:
            for scenario_name, scenario_prices in scenarios.items():
                scenario_results[scenario_name] = {
                    'oil_revenue': [p * scenario_prices['oil_price'] for p in oil_production],
                    'gas_revenue': [p * scenario_prices['gas_price'] for p in gas_production]
                }
                scenario_results[scenario_name]['total_revenue'] = [
                    o + g for o, g in zip(
                        scenario_results[scenario_name]['oil_revenue'],
                        scenario_results[scenario_name]['gas_revenue']
                    )
                ]
        
        return {
            "status": "success",
            "data": {
                "base_case": base_case,
                "scenarios": scenario_results
            },
            "metadata": {
                "total_oil_revenue": sum(base_case['oil_revenue']),
                "total_gas_revenue": sum(base_case['gas_revenue']),
                "total_revenue": sum(base_case['total_revenue'])
            }
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}

@tool
def calculate_costs(oil_production: list, gas_production: list, capex: float, fixed_opex: float, variable_opex: float, royalty_rate: float, transportation: float) -> Dict:
    """
    Calculate monthly cost components

    Parameters:
    - oil_production: Monthly oil production in bbl
    - gas_production: Monthly gas production in mcf
    - capex: Initial capital expenditure
    - fixed_opex: Fixed operating cost per month
    - variable_opex: Variable operating cost per BOE
    - royalty_rate: Royalty rate as a percentage of revenue
    - transportation: Transportation cost per BOE

    Returns a dictionary with monthly cost components
    """
    try:
        periods = len(oil_production)
        
        # Calculate variable costs based on total BOE
        total_boe = [oil + gas / 6 for oil, gas in zip(oil_production, gas_production)]
        variable_opex_costs = [boe * variable_opex for boe in total_boe]
        
        # Fixed costs array
        fixed_opex_costs = [fixed_opex] * periods
        
        # Total opex
        total_opex = [f + v for f, v in zip(fixed_opex_costs, variable_opex_costs)]
        
        # Transportation costs
        transportation_costs = [boe * transportation for boe in total_boe]
        
        # Capex (assumed to be incurred at start)
        capex_costs = [capex] + [0] * (periods - 1)
        
        cost_data = {
            'fixed_opex': fixed_opex_costs,
            'variable_opex': variable_opex_costs,
            'total_opex': total_opex,
            'transportation': transportation_costs,
            'capex': capex_costs
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
def calculate_monthly_cashflow(total_revenue: list, total_opex: list, transportation: list, capex: list, royalty_rate: float) -> Dict:
    """
    Calculate monthly cash flows

    Parameters:
    - total_revenue: Monthly total revenue
    - total_opex: Monthly total operating expenses
    - transportation: Monthly transportation costs
    - capex: Monthly capital expenditures
    - royalty_rate: Royalty rate as a percentage of revenue
    
    Returns a dictionary with monthly cash flows and operating cash flows
    """
    try:
        periods = len(total_revenue)
        
        # Calculate royalties
        royalties = [rev * royalty_rate for rev in total_revenue]
        
        # Calculate monthly cash flows
        cash_flows = []
        operating_cash_flows = []  # Before capex
        
        for i in range(periods):
            # Operating cash flow
            operating_cf = (
                total_revenue[i] -  # Revenue
                total_opex[i] -     # Opex
                transportation[i] - # Transportation
                royalties[i]        # Royalties
            )
            operating_cash_flows.append(operating_cf)
            
            # Total cash flow (including capex)
            total_cf = operating_cf - capex[i]
            cash_flows.append(total_cf)
        
        return {
            "status": "success",
            "data": {
                "operating_cash_flows": operating_cash_flows,
                "total_cash_flows": cash_flows,
                "royalties": royalties
            },
            "metadata": {
                "total_operating_cf": sum(operating_cash_flows),
                "total_cash_flow": sum(cash_flows),
                "total_royalties": sum(royalties)
            }
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}

@tool
def calculate_metrics(total_cash_flows: list, discount_rate: float = 0.10) -> Dict:
    """
    Calculate economic metrics using monthly cash flows

    Parameters:
    - total_cash_flows: Monthly total cash flows
    - discount_rate: Annual discount rate (default 10%)

    Returns a dictionary with NPV, IRR, payback period, and profitability index
    """
    try:
        # Convert to numpy array for calculations
        cf_array = np.array(total_cash_flows)
        
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
                "npv": npv,
                "monthly_irr": monthly_irr * 100 if monthly_irr is not None else None,
                "annual_irr": annual_irr * 100 if annual_irr is not None else None,
                "payback_period_months": payback_period,
                "payback_period_years": payback_period / 12 if payback_period is not None else None,
                "profitability_index": pi,
                "cumulative_cash_flow": cumulative_cf.tolist()
            },
            "metadata": {
                "annual_discount_rate": discount_rate,
                "monthly_discount_rate": monthly_rate,
                "analysis_period_months": len(cf_array)
            }
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}
    
tools = [calculate_production, calculate_revenue, calculate_costs, calculate_monthly_cashflow, calculate_metrics]