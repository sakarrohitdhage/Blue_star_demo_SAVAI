#!/usr/bin/env python3
"""
view_database.py - Utility script to view the contents of database tables

This script provides a simple command-line interface to view the contents of
database tables in the Sakar Vision AI database.
"""

import sys
import argparse
from azure_database import get_connection
from datetime import datetime, timedelta

def display_table_schema(table_name):
    """Display the schema of a table."""
    connection = get_connection()
    if not connection:
        print("Error: Could not connect to database")
        return False
        
    try:
        cursor = connection.cursor()
        cursor.execute(f"DESC {table_name}")
        
        # Print the column headers
        print(f"+{'-'*15}+{'-'*14}+{'-'*6}+{'-'*5}+{'-'*9}+{'-'*16}+")
        print(f"| {'Field':<15}| {'Type':<14}| {'Null':<6}| {'Key':<5}| {'Default':<9}| {'Extra':<16}|")
        print(f"+{'-'*15}+{'-'*14}+{'-'*6}+{'-'*5}+{'-'*9}+{'-'*16}+")
        
        # Print each row from the result
        for row in cursor.fetchall():
            field, type_val, null, key, default, extra = (str(item) if item is not None else "NULL" for item in row)
            print(f"| {field:<15}| {type_val:<14}| {null:<6}| {key:<5}| {default:<9}| {extra:<16}|")
        
        print(f"+{'-'*15}+{'-'*14}+{'-'*6}+{'-'*5}+{'-'*9}+{'-'*16}+")
        return True
            
    except Exception as e:
        print(f"Error retrieving table schema: {e}")
        return False
    finally:
        if connection.is_connected():
            cursor.close()
            connection.close()

def view_table_contents(table_name, limit=10, where_clause=None, order_by=None):
    """
    View the contents of a database table.
    
    Args:
        table_name (str): The name of the table to view
        limit (int): Maximum number of rows to display (default: 10)
        where_clause (str): Optional WHERE clause for filtering
        order_by (str): Optional ORDER BY clause for sorting
        
    Returns:
        bool: True if successful, False otherwise
    """
    connection = get_connection()
    if not connection:
        print("Error: Could not connect to database")
        return False
        
    try:
        cursor = connection.cursor(dictionary=True)
        
        # Construct the query with optional clauses
        query = f"SELECT * FROM {table_name}"
        if where_clause:
            query += f" WHERE {where_clause}"
        if order_by:
            query += f" ORDER BY {order_by}"
        query += f" LIMIT {limit}"
        
        print(f"Executing query: {query}")
        cursor.execute(query)
        results = cursor.fetchall()
        
        if not results:
            print("No results found.")
            return True
            
        # Get column names from the first result
        columns = list(results[0].keys())
        
        # Calculate column widths
        col_widths = {col: max(len(col), max([len(str(row[col])) for row in results])) for col in columns}
        
        # Print header row
        header = " | ".join([f"{col:<{col_widths[col]}}" for col in columns])
        separator = "-+-".join(["-" * col_widths[col] for col in columns])
        print(header)
        print(separator)
        
        # Print data rows
        for row in results:
            data_row = " | ".join([f"{str(row[col]):<{col_widths[col]}}" for col in columns])
            print(data_row)
            
        print(f"\n{len(results)} row(s) returned")
        return True
            
    except Exception as e:
        print(f"Error retrieving table contents: {e}")
        return False
    finally:
        if connection.is_connected():
            cursor.close()
            connection.close()

def main():
    """Parse command line arguments and execute the appropriate function."""
    parser = argparse.ArgumentParser(description="View database tables and contents")
    parser.add_argument("--table", "-t", help="Table name to view")
    parser.add_argument("--schema", "-s", action="store_true", help="Show table schema")
    parser.add_argument("--limit", "-l", type=int, default=10, help="Maximum rows to display")
    parser.add_argument("--where", "-w", help="WHERE clause for filtering")
    parser.add_argument("--order-by", "-o", help="ORDER BY clause for sorting")
    
    args = parser.parse_args()
    
    if args.schema and args.table:
        display_table_schema(args.table)
    elif args.table:
        view_table_contents(args.table, args.limit, args.where, args.order_by)
    else:
        parser.print_help()
    
if __name__ == "__main__":
    main()