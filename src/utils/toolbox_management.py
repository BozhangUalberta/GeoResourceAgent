# # searching
# from src.integrations.tools.test_websearcher_tools import websearch_tool

# geology
# from src.integrations.tools.las_reader_tools import las_reader
from src.integrations.tools.geostats_tools import geostats_interpolation, correlation_analysis, importance_analysis, plot_distribution
from src.integrations.tools.lasAnalysis_tools import auto_zonation
from src.integrations.tools.geopred_tools import geopred_conv_source

# Production
from src.integrations.tools.DCA_tools import DCA_tool_conversation_source, DCA_tool_table_source
from src.integrations.tools.plot_fitted_curve_tools import plot_fitted_curve_tool, plot_fitted_curve_tool_table_source
from src.integrations.tools.plot_table_curves_tools import smart_plot_production_curves_table_source, plot_curves_from_multiple_tables

# Plot tools
from src.integrations.tools.smart_plot import smart_plot_table_source, smart_plot_conversation_source

# Clean
from src.integrations.tools.data_parser import csv_to_database_table, las_to_database_table

# SQL
from src.integrations.tools.ProcessSQL.ProcessSQL_summarizer import ProcessSQL_summarizer
from src.integrations.tools.ProcessSQL.ProcessSQL_coder import ProcessSQL_coder
from src.integrations.tools.ProcessSQL.ProcessSQL_processor import ProcessSQL_processor
from src.integrations.tools.ProcessSQL.ProcessSQL_merger import ProcessSQL_merger
from src.integrations.tools.Table_search_and_process.table_deep_query import table_deep_query
from src.integrations.tools.Table_search_and_process.table_quick_peak import table_quick_peak

# Econ
# from src.integrations.tools.econ_tools import calculate_production, calculate_revenue, calculate_costs, calculate_monthly_cashflow, calculate_metrics
from src.integrations.tools.econ_tools_1205_revJZ import calculate_well_c_star,calculate_combined_royalty_rate,calculate_production,calculate_revenue,calculate_monthly_costs,calculate_monthly_cashflow,calculate_metrics
# from src.integrations.tools.econ_tools_1205_conversation import calculate_combined_royalty_rate_cov

# Accounting
from src.integrations.tools.calculate_basic.calculator import calculator

# RAG
from src.integrations.tools.RAG_tools import web_url_content_reader, pdf_reader,retriever

# ESG 
# from src.integrations.tools.Methane_Emission_tools import methane_analysis_pipeline
# from src.integrations.tools.plot_methane_emission_rate_tools import plot_methane_emission_rate
# from src.integrations.tools.CCUS_prediction_tools import predict_CCUS_Para
# from src.integrations.tools.Python_repl_tools import python_repl

# Geo agent tools
def define_geo_toolbox():
    geostats_tools = [geostats_interpolation, correlation_analysis, importance_analysis, plot_distribution]
    geopred_tools = [geopred_conv_source]
    lasAnalysis_tools = [auto_zonation]
    productionAnalysis_tools = [DCA_tool_conversation_source, DCA_tool_table_source, smart_plot_production_curves_table_source, plot_curves_from_multiple_tables]
    clean_tools = [csv_to_database_table, las_to_database_table]
    # sql_tools = [ProcessSQL_summarizer, ProcessSQL_coder, ProcessSQL_processor]#, ProcessSQL_merger]
    sql_tools = [table_quick_peak, table_deep_query]#, ProcessSQL_merger]
    # econ_tools = [calculate_well_c_star,calculate_combined_royalty_rate,calculate_production,calculate_revenue,calculate_monthly_costs,calculate_monthly_cashflow,calculate_metrics,smart_plot_table_source]

    rag_tools = [web_url_content_reader, pdf_reader]
    RAG_retriever_node = [retriever]

    # Combining all tool groups into a single toolbox
    toolbox = {
        "geostats_tools": geostats_tools,
        "geopred_tools": geopred_tools,
        "lasAnalysis_tools": lasAnalysis_tools,
        "productionAnalysis_tools": productionAnalysis_tools,
        "clean_tools": clean_tools,
        "sql_tools": sql_tools,
        # "econ_tools": econ_tools,
        # "rag_tools": rag_tools,
        # "RAG_retriever_node": RAG_retriever_node,
    }
     
    return toolbox

# Econ agent tools
def define_econ_toolbox():
    clean_tools = [csv_to_database_table]
    sql_tools = [ProcessSQL_summarizer, ProcessSQL_coder, ProcessSQL_processor]
    econ_tools = [calculate_well_c_star,calculate_combined_royalty_rate,calculate_production,calculate_revenue,calculate_monthly_costs,
                  calculate_monthly_cashflow,calculate_metrics,smart_plot_table_source]

    rag_tools = [web_url_content_reader, pdf_reader]
    RAG_retriever_node = [retriever]

    # Combining all tool groups into a single toolbox
    toolbox = {
        "clean_tools": clean_tools,
        "sql_tools": sql_tools,
        "econ_tools": econ_tools,
        "rag_tools": rag_tools,
        "RAG_retriever_node": RAG_retriever_node,
    }
     
    return toolbox

# Accounting Agent Tools
def define_accounting_toolbox():
    clean_tools = [csv_to_database_table]
    calculate_tools = [calculator]

    toolbox = {
        "clean_tools": clean_tools,
        "calculate_tools": calculate_tools,
    }

    return toolbox

# Production agent tools
def define_prod_toolbox():
    clean_tools = [csv_to_database_table]
    sql_tools = [table_quick_peak, table_deep_query, ProcessSQL_merger]
    productionAnalysis_tools = [DCA_tool_conversation_source, DCA_tool_table_source,
                                 smart_plot_production_curves_table_source,
                                   plot_curves_from_multiple_tables]

    rag_tools = [web_url_content_reader, pdf_reader]
    RAG_retriever_node = [retriever]

    # Combining all tool groups into a single toolbox
    toolbox = {
        "clean_tools": clean_tools,
        "sql_tools": sql_tools,
        "productionAnalysis_tools": productionAnalysis_tools,
        "rag_tools": rag_tools,
        "RAG_retriever_node": RAG_retriever_node,
    }
     
    return toolbox

# Econ agent tools
def define_ESG_toolbox():
    clean_tools = [csv_to_database_table]
    sql_tools = [table_quick_peak, table_deep_query]
    ESG_tools = [methane_analysis_pipeline, plot_methane_emission_rate, predict_CCUS_Para, python_repl]

    rag_tools = [web_url_content_reader, pdf_reader]
    RAG_retriever_node = [retriever]

    # Combining all tool groups into a single toolbox
    toolbox = {
        "clean_tools": clean_tools,
        "sql_tools": sql_tools,
        "ESG_tools": ESG_tools,
        "rag_tools": rag_tools,
        "RAG_retriever_node": RAG_retriever_node,
    }
     
    return toolbox