from langchain_core.tools import tool
from typing import Annotated, List, Literal, Dict
from langgraph.prebuilt import InjectedState

import numpy as np
import pandas as pd
import json
import os

from src.utils.db_utils import *

#####################
####### .EXCEL ######
#####################
@tool
def excel_file_parser(
    userID: Annotated[str, InjectedState("userID")],
    rowid: Annotated[str, f"User uploaded file location."],
    sheets: Annotated[List[int|str], f"str is certain sheet's exact name, int (starts from 0) is sheet order."] = None,
):
    """
    User uploaded excel files parser.
    It reads Excel like (.xls,.xlsx,.xlsm,.xlsb,.odf,.ods,.odt) into Pandas DataFrame.
    In most of times, this file should be provided by user, except user indicated some public sources that reachable to you.
    """
    try:
        data_path = get_data_sqlite3(filename=userID, table="running", id=rowid, type="userinput")
        
        file_name = os.path.basename(data_path)
        data_path = f"src/static/{userID}/{file_name}"

        if not os.path.exists(data_path):
            raise ValueError(f"File {data_path} does not exists.")

        # Read Excel as pd object
        excel = pd.ExcelFile(data_path)
        sheet_names = excel.sheet_names # List
        
        # Sheet selection check
        output = dict()
        output_sheet_names_shapes = dict()

        if sheets != None:
            for i, item in enumerate(sheets):
                if isinstance(item, int):
                    if item in range(len(sheet_names)):
                        output[sheet_names[item]] = excel.parse(item)
                        output_sheet_names_shapes[sheet_names[item]] = excel.parse(item).shape
                    else:
                        raise ValueError(f"int {item} in input_param::sheets[{i}] is over overall sheets length {len(sheet_names)}.")
                elif isinstance(item, str):
                    if item in sheet_names:
                        output[item] = excel.parse(item)
                        output_sheet_names_shapes[item] = excel.parse(item).shape
                    else:
                        raise ValueError(f"str {item} in input_param::sheets[{i}] is not found in sheets_names {sheet_names}.")


        description = f"Parsed user uploaded file with rowid={rowid} by function 'excel_file_parser'. Included sheets name and shape={output_sheet_names_shapes}."
        new_file_rowid = store_data_sqlite3(filename="test.db", table=userID, data=output, type="pddict", description=description)

        return json.dumps(f"User uploaded Excel file {data_path} with rowid={rowid} has been parsed as format: dict['sheet_name']=pddataframe into database with rowid={new_file_rowid}, names and shapes={output_sheet_names_shapes}.")
    
    except Exception as e:
        return json.dumps(f"Tool 'excel_file_parser' wrong with userID='{userID}', path='{data_path}' and rowid={rowid}, here is error info: {e}")
    



@tool
def list_and_dict_parser(
    userID: Annotated[str, InjectedState("userID")],
    data_body: Annotated[list|dict, f"Data that need to be stored into database running table."],
    description: Annotated[str, f"Detailed description for this data record, include necessary information so you can understand it in future usage."],
):
    """
    User prompt that contains possible python List or Dictionary(or json) should use this function to parse.
    If user prompt contains multiple lists or Dictionaries(or json), use this function multiple times to parse all inputs.
    """
    try:
        if isinstance(data_body, list):
            data_type = "list"
        else:
            data_type = "dict"

        new_rowid = store_data_sqlite3(filename=userID, table="running", data=data_body, type=data_type, description=description)
        
        return json.dumps(f"User prompt input {data_type} data and content={data_body} has been stored with rowid={new_rowid}.")
    
    except Exception as e:
        return json.dumps(f"Tool 'list_and_dict_parser' error with data={data_body}. Here is error info: {e}")


#####################
####### .CSV ########
#####################
import pandas as pd
import sqlite3
@tool
def csv_to_database_table(
    userID: Annotated[str, InjectedState("userID")],
    row_id: Annotated[int, f"Row ID for csv file location in 'running' table, it should be a user-uploaded file."],
    new_table_name: Annotated[str, f"Use CSV file name without extension type, modify it to fit SQLite table name rules."],
):
    """
    Convert user uploaded **CSV file** into a new SQLite data table, preserving original data types.
    """
    try:
        # Connect to the SQLite database (creates it if it doesn't exist)
        file_path = get_data_sqlite3(filename=userID, table="running", id=row_id, type="userinput")
        db_name = os.path.join("database", userID + ".db")

        # Read the CSV file using pandas
        df = pd.read_csv(file_path)

        # Replace NaNs with None for SQLite compatibility
        df = df.replace({np.nan: None})

        # Attempt to convert object columns to numeric
        for column in df.columns:
            if df[column].dtype == object:
                try:
                    df[column] = pd.to_numeric(df[column])
                except ValueError:
                    pass  # Ignore if conversion fails

        # Helper function to infer SQLite-compatible types
        def infer_sql_type(series):
            try:
                pd.to_numeric(series.dropna())
                if pd.api.types.is_integer_dtype(series):
                    return "INTEGER"
                elif pd.api.types.is_float_dtype(series):
                    return "REAL"
            except ValueError:
                pass
            if pd.api.types.is_bool_dtype(series):
                return "BOOLEAN"
            elif pd.api.types.is_datetime64_any_dtype(series):
                return "DATETIME"
            else:
                return "TEXT"

        # Infer column types
        column_definitions = []
        for column in df.columns:
            inferred_type = infer_sql_type(df[column])
            column_definitions.append(f'"{column}" {inferred_type}')

        # Generate SQL for table creation
        create_table_query = f'CREATE TABLE IF NOT EXISTS "{new_table_name}" ({", ".join(column_definitions)});'

        # Connect to the SQLite database
        conn = sqlite3.connect(db_name)
        cursor = conn.cursor()

        # Create the table
        cursor.execute(create_table_query)

        # Insert the DataFrame into the SQLite table
        df.to_sql(new_table_name, conn, if_exists='append', index=False)

        # Commit and close the connection
        conn.commit()
        conn.close()

        return json.dumps(f"New table '{new_table_name}' has been created successfully with data from CSV.")
    
    except Exception as e:
        return json.dumps(f"Tool 'csv_to_database_table' error: {e}")


#####################
####### .LAS ########
#####################
import lasio
import uuid

@tool
def las_to_database_table(
    userID: Annotated[str, InjectedState("userID")],
    row_id: Annotated[int, f"Row ID for LAS file location in 'running' table, it should be a user-uploaded file."], 
    file_name: Annotated[str, "The original file name"], 
):
    """
    Convert a user-uploaded **LAS file** into a new data table.
    Now supports handling of missing values and SQLite-compatible data types.
    """
    print("========The .LAS reader is used=========")
    try:
        # Connect to the SQLite database (creates it if it doesn't exist)
        file_path = get_data_sqlite3(filename=userID, table="running", id=row_id, type="userinput")
        db_name = os.path.join("database", userID + ".db")

        # Read the LAS file using lasio
        las = lasio.read(file_path, engine="normal")
        df = pd.DataFrame(las.data, columns=las.keys())

        df.dropna(axis=1, how='all', inplace=True)

        # Step 2: Fill NaN values according to the rules
        for column in df.columns:
            # Check if the first value is NaN, and if so, fill it with the first available value in that column
            if pd.isna(df[column].iloc[0]):
                first_valid_index = df[column].first_valid_index()
                if first_valid_index is not None:
                    df.loc[0, column] = df[column].iloc[first_valid_index]
            
            # Forward fill the remaining NaN values using the previous valid value
            df[column] = df[column].ffill()

        # Convert numpy data types (e.g., NaNs) for SQLite compatibility
        df = df.replace({np.nan: None})  # Replace NaN with None for SQLite compatibility

        # Convert columns with numpy object types (if any) into a standard type for SQLite (e.g., TEXT)
        df = df.apply(lambda x: str(x) if isinstance(x, (np.ndarray, np.generic)) else x)

        # Connect to the SQLite database
        conn = sqlite3.connect(db_name)
        cursor = conn.cursor()

        # Infer data types from the DataFrame
        new_table_name= f"LAS_{uuid.uuid4().hex[:6]}"
        columns = ', '.join([f'"{col}" TEXT' for col in df.columns])  # Assuming TEXT for simplicity
        create_table_query = f'CREATE TABLE IF NOT EXISTS {new_table_name} ({columns})'
        cursor.execute(create_table_query)

        # Insert the DataFrame into the SQLite table
        df.to_sql(new_table_name, conn, if_exists='append', index=False)

        # Commit and close the connection
        conn.commit()
        conn.close()

        return json.dumps(f"File {file_name} is saved as a New table {new_table_name}.")
    
    except Exception as e:
        return json.dumps(f"Tool 'las_to_database_table' error: {e}")