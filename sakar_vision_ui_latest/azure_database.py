#!/usr/bin/env python3
"""
azure_database.py - Azure database connection and operations

This module provides functions to connect to and interact with an Azure MySQL database
for user authentication. It implements secure connection patterns, connection pooling,
and parameterized queries to prevent SQL injection.
"""

import os
import json
import logging
import mysql.connector
from mysql.connector import pooling
from datetime import datetime
import time
import random
import uuid  # For generating unique IDs
import hashlib  # For local authentication

# Path to local user file for fallback authentication
USERS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "users.json")

# Session storage to track pending location entries
_pending_location_entries = {}

# Circuit breaker state for database connections
_circuit_breaker = {
    "failures": 0,
    "threshold": 5,  # Number of failures before circuit opens
    "reset_time": None,  # When to try resetting the circuit
    "reset_timeout": 300,  # Seconds to wait before trying to reset (5 minutes)
    "is_open": False  # Whether the circuit is open
}

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('azure_database')

# Configuration
# In production, these should be retrieved from Azure Key Vault or environment variables
DB_CONFIG = {
    "host": os.environ.get("AZURE_MYSQL_HOST", "metal-sheet-db.mysql.database.azure.com"),
    "user": os.environ.get("AZURE_MYSQL_USER", "sakarmetal"),
    "password": os.environ.get("AZURE_MYSQL_PASSWORD", "Sakar@321"),
    "database": os.environ.get("AZURE_MYSQL_DATABASE", "metal_sheet"),
    "port": int(os.environ.get("AZURE_MYSQL_PORT", "3306")),
    "ssl_ca": os.environ.get("AZURE_MYSQL_SSL_CA", None),  # Path to SSL CA certificate
    "ssl_verify_cert": False,
    "connection_timeout": 10
}

# Create a connection pool for better performance
try:
    connection_pool = pooling.MySQLConnectionPool(
        pool_name="azure_mysql_pool",
        pool_size=5,  # Adjust based on expected load
        **DB_CONFIG
    )
    logger.info("Connection pool created successfully")
except Exception as e:
    logger.error(f"Error creating connection pool: {e}")
    connection_pool = None

def validate_username(username):
    """
    Validate username before database lookup to prevent "unknown" user queries.
    
    Args:
        username (str): Username to validate
        
    Returns:
        bool: True if username is valid for database lookup, False otherwise
    """
    if not username:
        return False
    
    # Convert to string if not already
    username = str(username)
    
    # Invalid usernames that should not be looked up
    invalid_usernames = ['unknown', 'null', 'none', '', 'undefined', 'test']
    
    if username.lower() in invalid_usernames:
        logger.warning(f"Rejecting invalid username for database lookup: '{username}'")
        return False
    
    # Check if username looks like a dictionary representation
    if username.startswith('{') and username.endswith('}'):
        logger.warning(f"Rejecting dictionary-like username: '{username}'")
        return False
    
    return True

# Then update the get_user_id_by_username function to use this validation:

def get_user_id_by_username(username):
    """
    Get the user ID for a specific username from the users table.
    When multiple rows with the same username exist, consistently returns the one with the lowest ID.

    Args:
        username (str): The username to look up

    Returns:
        str: User ID if found, None otherwise
    """
    # Validate username before proceeding
    if not validate_username(username):
        logger.info(f"Skipping database lookup for invalid username: {username}")
        return None
    
    connection = get_connection()
    if not connection:
        logger.error(f"Cannot get user ID: No database connection")
        return None

    try:
        cursor = connection.cursor(dictionary=True, buffered=True)

        # Modified query to sort by ID when multiple matches exist
        # ORDER BY id ensures we consistently get the same ID for the same username
        query = "SELECT id FROM users WHERE username = %s ORDER BY id ASC LIMIT 1"
        cursor.execute(query, (username,))
        user = cursor.fetchone()

        if user:
            logger.info(f"Found user ID {user['id']} for username {username}")
            return user['id']

        # Try case-insensitive query as fallback
        query = "SELECT id FROM users WHERE LOWER(username) = LOWER(%s) ORDER BY id ASC LIMIT 1"
        cursor.execute(query, (username,))
        user = cursor.fetchone()

        if user:
            logger.info(
                f"Found user ID {user['id']} for username {username} using case-insensitive search")
            return user['id']

        logger.info(f"User not found: {username}")
        return None

    except mysql.connector.Error as err:
        # Handle specific MySQL errors
        if err.errno == 1146:  # Table doesn't exist
            logger.error("Table 'users' does not exist in the database")
            return None
        elif err.errno == 1054:  # Unknown column
            logger.warning(f"Column error in users table: {err}. Trying fallback query.")

            # Try a more generic approach to get whatever columns are available
            cursor.close()
            cursor = connection.cursor(dictionary=True, buffered=True)

            try:
                # Get table structure
                cursor.execute("DESCRIBE users")
                columns = [row['Field'] for row in cursor.fetchall()]

                # Find id column and username-like columns
                id_col = next((col for col in columns if col.lower() == 'id'), None)
                name_cols = [col for col in columns if 'user' in col.lower()
                             or 'name' in col.lower()]

                if id_col and name_cols:
                    # Use the first name-like column we found
                    name_col = name_cols[0]
                    # Construct a dynamic query with the actual column names and sort by ID
                    query = f"SELECT {id_col} FROM users WHERE {name_col} = %s ORDER BY {id_col} ASC LIMIT 1"
                    cursor.execute(query, (username,))
                    user = cursor.fetchone()

                    if user:
                        user_id = user[id_col]
                        logger.info(
                            f"Found user ID {user_id} for username {username} using column {name_col}")
                        return user_id

                logger.error(
                    f"Could not find user with username {username} using available columns")
                return None

            except Exception as col_err:
                logger.error(f"Error in fallback column approach: {col_err}")
                return None
        else:
            # Re-raise other database errors
            raise

    except Exception as e:
        logger.error(f"Error getting user ID: {e}")
        # Provide more details about the error for debugging
        import traceback
        logger.debug(f"Traceback: {traceback.format_exc()}")
        return None
    finally:
        if connection.is_connected():
            cursor.close()
            connection.close()
            
def check_circuit_breaker():
    """
    Check if the circuit breaker is open, and if it should be reset.

    Returns:
        bool: True if circuit is closed (connections allowed), False if open (connections not allowed)
    """
    if not _circuit_breaker["is_open"]:
        return True

    # Check if we should attempt to reset the circuit
    if _circuit_breaker["reset_time"] and datetime.now().timestamp() > _circuit_breaker["reset_time"]:
        logger.info("Attempting to reset circuit breaker")
        _circuit_breaker["is_open"] = False
        _circuit_breaker["failures"] = 0
        return True

    logger.warning("Circuit breaker is open, blocking database connection attempts")
    return False


def record_connection_failure():
    """Record a connection failure and potentially open the circuit breaker."""
    _circuit_breaker["failures"] += 1

    if _circuit_breaker["failures"] >= _circuit_breaker["threshold"]:
        if not _circuit_breaker["is_open"]:
            logger.critical(
                f"Circuit breaker triggered after {_circuit_breaker['failures']} failures")
            _circuit_breaker["is_open"] = True

        # Set a reset time to attempt recovery after the timeout
        _circuit_breaker["reset_time"] = datetime.now().timestamp() + \
            _circuit_breaker["reset_timeout"]


def record_connection_success():
    """Record a successful connection and reset the failure counter."""
    if (_circuit_breaker["failures"] > 0):
        _circuit_breaker["failures"] = 0
        logger.info("Connection successful, reset failure counter")

    if _circuit_breaker["is_open"]:
        _circuit_breaker["is_open"] = False
        logger.info("Circuit breaker reset after successful connection")


def get_connection(use_circuit_breaker=True):
    """
    Get a connection with enhanced retry logic and circuit breaker pattern.

    Args:
        use_circuit_breaker (bool): Whether to use the circuit breaker pattern

    Returns:
        MySQL connection object or None if connection fails
    """
    # Check circuit breaker to prevent repeated connection attempts
    if use_circuit_breaker and not check_circuit_breaker():
        logger.warning("Connection attempt blocked by circuit breaker")
        return None

    max_retries = 3
    retry_count = 0

    while retry_count < max_retries:
        try:
            # Implement exponential backoff with jitter for retries
            if retry_count > 0:
                # Calculate delay: 2^retry_count seconds + random jitter
                delay = (2 ** retry_count) + (random.randint(0, 1000) / 1000.0)
                logger.info(
                    f"Retry attempt {retry_count} for database connection, waiting {delay:.2f} seconds")
                time.sleep(delay)

            if connection_pool:
                connection = connection_pool.get_connection()
                # Test connection is actually working with a simple query
                cursor = connection.cursor()
                cursor.execute("SELECT 1")
                cursor.fetchone()
                cursor.close()

                # Record success and return connection
                record_connection_success()
                return connection
            else:
                # Fallback to direct connection if pool is not available
                logger.warning("Using direct connection instead of pool")
                connection = mysql.connector.connect(**DB_CONFIG)

                # Test connection is working
                cursor = connection.cursor()
                cursor.execute("SELECT 1")
                cursor.fetchone()
                cursor.close()

                # Record success and return connection
                record_connection_success()
                return connection

        except Exception as e:
            retry_count += 1
            logger.error(f"Connection attempt {retry_count} failed: {e}")

            # Record the failure for circuit breaker
            if use_circuit_breaker:
                record_connection_failure()

            # Special handling for MySQL connection errors
            if isinstance(e, mysql.connector.Error):
                if e.errno == 2003:  # Can't connect to MySQL server
                    logger.error(f"MySQL Server Connection Error: {str(e)}")
                elif e.errno == 1045:  # Access denied
                    logger.error(f"MySQL Authentication Error: {str(e)}")
                    # If authentication failed, no need to retry
                    break

    logger.critical("Failed to connect to Azure MySQL database after multiple attempts")
    return None


def authenticate_user(username, password):
    """
    Authenticate a user against the Azure MySQL database with local fallback.
    Uses case-sensitive comparison for both username and password.

    Args:
        username (str): The username to check
        password (str): The password to check

    Returns:
        str: User ID if authentication is successful, None otherwise
    """
    # Try Azure database authentication first
    user_id = authenticate_user_with_db(username, password)
    if user_id:
        return user_id

    # If that fails, try local authentication
    user_from_local = authenticate_user_local(username, password)
    if user_from_local:
        logger.info(f"User authenticated locally: {username}")
        return user_from_local.get('id')

    # If both fail, authentication is unsuccessful
    logger.info(f"Authentication failed for user: {username}")
    return None


def authenticate_user_with_db(username, password):
    """
    Authenticate user against the Azure MySQL database.
    When multiple rows with the same username exist, uses the lowest ID consistently.

    Args:
        username (str): The username to check
        password (str): The password to check

    Returns:
        str: User ID if authentication is successful, None otherwise
    """
    connection = get_connection()
    if not connection:
        logger.error("Cannot authenticate: No database connection")
        return None

    try:
        cursor = connection.cursor(dictionary=True)

        # Modified query to sort by ID to ensure consistency with get_user_id_by_username
        # This ensures that for duplicate usernames, we always authenticate against the same record
        query = "SELECT * FROM users WHERE username = %s ORDER BY id ASC"
        cursor.execute(query, (username,))
        users = cursor.fetchall()

        if users:
            # Check if any user's password matches, prioritizing records with lower IDs
            for user in users:
                if user['password'] == password:  # Case-sensitive password comparison
                    logger.info(f"User authenticated successfully: {username} with ID: {user['id']}")
                    return user['id']
            
            logger.info(f"Authentication failed: Invalid password for user: {username}")
            return None
        else:
            logger.info(f"Authentication failed: User not found: {username}")
            return None

    except Exception as e:
        logger.error(f"Authentication error: {e}")
        return None
    finally:
        if connection.is_connected():
            cursor.close()
            connection.close()


def authenticate_user_local(username, password):
    """
    Fallback local authentication when Azure DB is unavailable.

    Args:
        username (str): The username to check
        password (str): The password to check

    Returns:
        dict: User information if authentication is successful, None otherwise
    """
    try:
        # For certain special accounts, check hardcoded credentials
        # (This is for emergency access only, should be removed in production)
        if username == "admin" and password == "admin123":
            return {
                "id": "A001",
                "username": "admin",
                "password": "admin123"  # In a real system, never return the password
            }

        # Otherwise check against users.json
        if os.path.exists(USERS_FILE):
            with open(USERS_FILE, 'r') as f:
                users = json.load(f)

            # Check if user exists
            if username in users:
                # Check password - handle both hashed and plain password formats
                user_data = users[username]

                if "password_hash" in user_data:
                    # Hashed password format
                    stored_hash = user_data["password_hash"]
                    provided_hash = hashlib.sha256(password.encode()).hexdigest()
                    if stored_hash == provided_hash:
                        # Convert from local format to database format
                        return {
                            # Generate an ID if none exists
                            "id": user_data.get("id", f"L{random.randint(100, 999)}"),
                            "username": username,
                            "password": password  # In a real system, never return the password
                        }
                elif "password" in user_data:
                    # Plain text password format (not recommended but supported for compatibility)
                    if user_data["password"] == password:
                        return {
                            "id": user_data.get("id", f"L{random.randint(100, 999)}"),
                            "username": username,
                            "password": password
                        }

        logger.info(f"Local authentication failed for user: {username}")
        return None
    except Exception as e:
        logger.error(f"Local authentication error: {e}")
        return None


def get_all_users():
    """
    Get all users from the database with enhanced error handling and local fallback.

    Returns:
        list: List of user dictionaries if successful, empty list otherwise
    """
    # Try to get users from database first
    db_users = get_all_users_from_db()
    if db_users:
        return db_users

    # If database access fails, try to get users from local file
    local_users = get_all_users_from_local()
    if local_users:
        logger.info(f"Returning {len(local_users)} users from local file instead of database")
        return local_users

    # If both fail, return empty list
    return []


def get_all_users_from_db():
    """
    Get all users from the database.

    Returns:
        list: List of user dictionaries if successful, empty list otherwise
    """
    connection = get_connection()
    if not connection:
        logger.error("Cannot get users: No database connection")
        return []

    try:
        cursor = connection.cursor(dictionary=True, buffered=True)

        # First try with expected column names
        try:
            query = "SELECT id, username FROM users"
            cursor.execute(query)
            users = cursor.fetchall()

            # Validate that the returned data has the expected structure
            if users and len(users) > 0:
                if 'id' not in users[0] or 'username' not in users[0]:
                    logger.warning(
                        "Missing expected columns in users table result. Trying fallback query.")
                    raise Exception("Missing expected columns")

            logger.info(f"Retrieved {len(users)} users")
            return users

        except mysql.connector.Error as err:
            # Handle specific MySQL errors
            if err.errno == 1146:  # Table doesn't exist
                logger.error("Table 'users' does not exist in the database")
                return []
            elif err.errno == 1054:  # Unknown column
                logger.warning(f"Column error in users table: {err}. Trying fallback query.")
                # Try a more generic approach to get whatever columns are available
                cursor.close()
                cursor = connection.cursor(dictionary=True, buffered=True)
                cursor.execute("DESCRIBE users")
                columns = [row['Field'] for row in cursor.fetchall()]

                # Check if we have at least an ID column and something that might be username
                id_col = next((col for col in columns if col.lower() == 'id'), None)
                username_cols = [col for col in columns if 'user' in col.lower()
                                 or 'name' in col.lower()]

                if id_col and username_cols:
                    username_col = username_cols[0]  # Take the first likely username column
                    query = f"SELECT {id_col}, {username_col} FROM users"
                    cursor.execute(query)
                    users = cursor.fetchall()

                    # Map columns to expected structure
                    standardized_users = []
                    for user in users:
                        standardized_users.append({
                            'id': user[id_col],
                            'username': user[username_col]
                        })

                    logger.info(
                        f"Retrieved {len(standardized_users)} users with fallback column mapping")
                    return standardized_users
                else:
                    logger.error("Could not identify required columns in users table")
                    return []
            else:
                # Re-raise other database errors
                raise

    except Exception as e:
        logger.error(f"Error retrieving users: {e}")
        # Provide more details about the error for debugging
        import traceback
        logger.debug(f"Traceback: {traceback.format_exc()}")
        return []
    finally:
        if connection.is_connected():
            cursor.close()
            connection.close()


def get_all_users_from_local():
    """
    Get all users from the local users.json file.

    Returns:
        list: List of user dictionaries if successful, empty list otherwise
    """
    try:
        if os.path.exists(USERS_FILE):
            with open(USERS_FILE, 'r') as f:
                users_dict = json.load(f)

            # Convert dictionary format to list format matching database output
            users_list = []
            for username, user_data in users_dict.items():
                # Create a user object with at least id and username
                user_obj = {
                    # Generate ID if none exists
                    'id': user_data.get("id", f"L{random.randint(100, 999)}"),
                    'username': username
                }
                users_list.append(user_obj)

            logger.info(f"Retrieved {len(users_list)} users from local file")
            return users_list
        else:
            logger.warning(f"Local users file not found: {USERS_FILE}")
            return []
    except Exception as e:
        logger.error(f"Error retrieving users from local file: {e}")
        return []


def test_connection():
    """
    Test the database connection with enhanced error reporting.

    Returns:
        bool: True if connection successful, False otherwise
    """
    connection = get_connection()
    if not connection:
        logger.error("Connection test failed: Could not get connection")
        return False

    try:
        cursor = connection.cursor()
        cursor.execute("SELECT 1")
        result = cursor.fetchone()

        if result and result[0] == 1:
            logger.info("Connection test successful")
            return True
        else:
            logger.error("Connection test failed: Unexpected result")
            return False

    except Exception as e:
        logger.error(f"Connection test failed: {e}")
        return False
    finally:
        if connection.is_connected():
            cursor.close()
            connection.close()


def store_organization_data(user_id, organization):
    """
    Store organization data in memory for later use.
    This function is called from metal_sheet_organization_ui.py when a user selects an organization.

    Args:
        user_id (str): The user ID from the users table
        organization (str): The selected organization name

    Returns:
        bool: True if data was stored successfully, False otherwise
    """
    try:
        # Store in memory for later use with store_location_data
        _pending_location_entries[user_id] = {
            'organization': organization,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }

        logger.info(f"Stored organization data in memory for user {user_id}: {organization}")
        return True
    except Exception as e:
        logger.error(f"Error storing organization data in memory: {e}")
        return False


def store_location_data(user_id, organization, city, project):
    """
    Store complete location data in the Azure MySQL database.
    Prevents duplicate entries of the same user_id, organization, city, project combination.

    If this is called from the second page (with city and project),
    it will use the organization previously stored with store_organization_data.

    Args:
        user_id (str or int): The ID of the logged-in user
        organization (str): The selected organization (or None if from second page)
        city (str): The selected city
        project (str): The selected project

    Returns:
        bool: True if data was stored successfully, False otherwise
    """
    # Skip temporary storage and partial database entries
    if city == "pending" and project == "pending":
        # This is the first page submission - store temporarily and return
        return store_organization_data(user_id, organization)

    connection = get_connection()
    if not connection:
        logger.error("Cannot store location data: No database connection")
        return False

    try:
        cursor = connection.cursor()

        # Use the current timestamp for the datetime field
        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        # Get organization from in-memory storage if not provided
        org_to_use = organization
        if org_to_use == "pending" or not org_to_use:
            if user_id in _pending_location_entries:
                org_to_use = _pending_location_entries[user_id]['organization']
                # Get timestamp from storage if available
                if 'timestamp' in _pending_location_entries[user_id]:
                    current_time = _pending_location_entries[user_id]['timestamp']
            else:
                logger.warning(f"No organization found for user {user_id}, using provided value")

        # Check if a record with the same organization, city, and project already exists, regardless of user_id
        # This ensures we don't create duplicates when multiple users with the same username try to create the same resource
        check_query = """
        SELECT location_id, id FROM location 
        WHERE organization = %s AND city = %s AND project = %s
        """
        cursor.execute(check_query, (org_to_use, city, project))
        existing_record = cursor.fetchone()

        if existing_record:
            # Record already exists, log and return success without inserting duplicate
            logger.info(
                f"Location data already exists for organization {org_to_use}, city {city}, project {project}. Skipping duplicate insertion.")

            # Clear the temporary storage
            if user_id in _pending_location_entries:
                del _pending_location_entries[user_id]

            return True

        # Insert a new record with the complete information
        insert_query = """
        INSERT INTO location (id, organization, city, project, timestamp) 
        VALUES (%s, %s, %s, %s, %s)
        """
        cursor.execute(insert_query, (user_id, org_to_use, city, project, current_time))
        connection.commit()

        # Get the auto-generated location_id
        location_id = cursor.lastrowid

        # Clear the temporary storage
        if user_id in _pending_location_entries:
            del _pending_location_entries[user_id]

        logger.info(
            f"Stored complete location data with location_id {location_id} for user {user_id}: {org_to_use}, {city}, {project}")
        return True

    except Exception as e:
        logger.error(f"Error storing location data: {e}")
        connection.rollback()
        logger.error(f"Detailed error: {str(e)}")
        return False
    finally:
        if connection.is_connected():
            cursor.close()
            connection.close()


def get_user_id_by_username(username):
    """
    Get the user ID for a specific username from the users table.
    When multiple rows with the same username exist, consistently returns the one with the lowest ID.

    Args:
        username (str): The username to look up

    Returns:
        str: User ID if found, None otherwise
    """
    connection = get_connection()
    if not connection:
        logger.error(f"Cannot get user ID: No database connection")
        return None

    try:
        cursor = connection.cursor(dictionary=True, buffered=True)

        # Modified query to sort by ID when multiple matches exist
        # ORDER BY id ensures we consistently get the same ID for the same username
        query = "SELECT id FROM users WHERE username = %s ORDER BY id ASC LIMIT 1"
        cursor.execute(query, (username,))
        user = cursor.fetchone()

        if user:
            logger.info(f"Found user ID {user['id']} for username {username}")
            return user['id']

        # Try case-insensitive query as fallback
        query = "SELECT id FROM users WHERE LOWER(username) = LOWER(%s) ORDER BY id ASC LIMIT 1"
        cursor.execute(query, (username,))
        user = cursor.fetchone()

        if user:
            logger.info(
                f"Found user ID {user['id']} for username {username} using case-insensitive search")
            return user['id']

        logger.info(f"User not found: {username}")
        return None

    except mysql.connector.Error as err:
        # Handle specific MySQL errors
        if err.errno == 1146:  # Table doesn't exist
            logger.error("Table 'users' does not exist in the database")
            return None
        elif err.errno == 1054:  # Unknown column
            logger.warning(f"Column error in users table: {err}. Trying fallback query.")

            # Try a more generic approach to get whatever columns are available
            cursor.close()
            cursor = connection.cursor(dictionary=True, buffered=True)

            try:
                # Get table structure
                cursor.execute("DESCRIBE users")
                columns = [row['Field'] for row in cursor.fetchall()]

                # Find id column and username-like columns
                id_col = next((col for col in columns if col.lower() == 'id'), None)
                name_cols = [col for col in columns if 'user' in col.lower()
                             or 'name' in col.lower()]

                if id_col and name_cols:
                    # Use the first name-like column we found
                    name_col = name_cols[0]
                    # Construct a dynamic query with the actual column names and sort by ID
                    query = f"SELECT {id_col} FROM users WHERE {name_col} = %s ORDER BY {id_col} ASC LIMIT 1"
                    cursor.execute(query, (username,))
                    user = cursor.fetchone()

                    if user:
                        user_id = user[id_col]
                        logger.info(
                            f"Found user ID {user_id} for username {username} using column {name_col}")
                        return user_id

                logger.error(
                    f"Could not find user with username {username} using available columns")
                return None

            except Exception as col_err:
                logger.error(f"Error in fallback column approach: {col_err}")
                return None
        else:
            # Re-raise other database errors
            raise

    except Exception as e:
        logger.error(f"Error getting user ID: {e}")
        # Provide more details about the error for debugging
        import traceback
        logger.debug(f"Traceback: {traceback.format_exc()}")
        return None
    finally:
        if connection.is_connected():
            cursor.close()
            connection.close()


def get_location_history(user_id, limit=10):
    """
    Get the location history for a user from the database.

    Args:
        user_id (str or int): The ID of the user to get history for
        limit (int): Maximum number of records to return (default: 10)

    Returns:
        list: List of location records sorted by timestamp DESC (most recent first)
    """
    connection = get_connection()
    if not connection:
        logger.error("Cannot get location history: No database connection")
        return []

    try:
        cursor = connection.cursor(dictionary=True)

        # Get location history sorted by timestamp (newest first)
        query = """
        SELECT location_id, id, organization, city, project, timestamp 
        FROM location 
        WHERE id = %s 
        ORDER BY timestamp DESC 
        LIMIT %s
        """
        cursor.execute(query, (user_id, limit))
        locations = cursor.fetchall()

        logger.info(f"Retrieved {len(locations)} location records for user {user_id}")
        return locations

    except Exception as e:
        logger.error(f"Error retrieving location history: {e}")
        return []
    finally:
        if connection.is_connected():
            cursor.close()
            connection.close()


def check_user_exists(username):
    """
    Checks if a user with the specified username exists in the users table.

    Args:
        username (str): The username to check

    Returns:
        bool: True if user exists, False otherwise
    """
    connection = get_connection()
    if not connection:
        logger.error(f"Cannot check if user '{username}' exists: No database connection")
        return False

    try:
        cursor = connection.cursor(dictionary=True)

        # Check if user exists using parameterized query
        query = "SELECT id FROM users WHERE username = %s"
        cursor.execute(query, (username,))
        user = cursor.fetchone()

        if user:
            logger.info(f"User '{username}' exists with ID: {user['id']}")
            return True
        else:
            logger.info(f"User '{username}' does not exist in the database")
            return False

    except Exception as e:
        logger.error(f"Error checking if user '{username}' exists: {e}")
        return False
    finally:
        if connection and connection.is_connected():
            cursor.close()
            connection.close()


def update_defect_count(defect, count, threshold=None, is_initialization=False, user_id=None, location_id=None):
    """
    Update the defect count in the Azure MySQL database.
    Counts are always attributed to the current date and location.

    Args:
        defect (str): The name of the defect.
        count (int): The count of the defect.
        threshold (int, optional): The threshold value for the defect.
        is_initialization (bool): Whether this update is part of initialization (don't increment).
        user_id (str, optional): The user ID to associate with this defect record.
        location_id (int, optional): The location ID to associate with this defect record.

    Returns:
        bool: True if the update was successful, False otherwise.
    """
    logger.info(
        f"DEBUG: update_defect_count called with defect={defect}, count={count}, threshold={threshold}, "
        f"is_initialization={is_initialization}, user_id={user_id}, location_id={location_id}")

    connection = get_connection()
    if not connection:
        logger.error("Cannot update defect count: No database connection")
        return False

    try:
        # Use buffered cursor to avoid unread result errors
        cursor = connection.cursor(buffered=True)

        # Get the current date in YYYY-MM-DD format
        current_date = datetime.now().strftime('%Y-%m-%d')
        # For database queries: start and end of the current day
        today_start = f"{current_date} 00:00:00"
        today_end = f"{current_date} 23:59:59"

        # Generate the current timestamp including time
        current_timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        # If location_id is not provided, try to fetch it
        if location_id is None and user_id:
            location_id = get_latest_location_id(user_id)
            logger.info(f"DEBUG: Retrieved location_id={location_id} for user_id={user_id}")

        logger.info(f"DEBUG: Looking for records for today's date: {current_date}")

        # Handle initialization differently - we want to add the class to database but not increment
        if is_initialization:
            # Default to admin user if no user_id specified
            init_user_id = user_id if user_id else "A001"

            # Check if this defect class already exists today for this user AND location
            if location_id:
                check_query = """
                SELECT structural_id FROM defects 
                WHERE defect = %s AND id = %s AND location_id = %s AND timestamp BETWEEN %s AND %s
                """
                cursor.execute(check_query, (defect, init_user_id,
                               location_id, today_start, today_end))
            else:
                check_query = """
                SELECT structural_id FROM defects 
                WHERE defect = %s AND id = %s AND timestamp BETWEEN %s AND %s
                """
                cursor.execute(check_query, (defect, init_user_id, today_start, today_end))

            today_record = cursor.fetchone()

            # If no record exists for today and this location, create a new one with count=0
            if not today_record:
                logger.info(
                    f"DEBUG: Initializing new class {defect} for user {init_user_id} with count=0 and location_id={location_id}")

                # Include location_id in the insert query if available
                if location_id:
                    insert_query = """
                    INSERT INTO defects (id, defect, count, threshold, timestamp, location_id) 
                    VALUES (%s, %s, %s, %s, %s, %s)
                    """
                    cursor.execute(insert_query, (init_user_id, defect, 0,
                                   threshold, current_timestamp, location_id))
                else:
                    insert_query = """
                    INSERT INTO defects (id, defect, count, threshold, timestamp) 
                    VALUES (%s, %s, %s, %s, %s)
                    """
                    cursor.execute(insert_query, (init_user_id, defect,
                                   0, threshold, current_timestamp))

                connection.commit()
                logger.info(f"DEBUG: Initialization successful for defect: {defect}")
                return True
            else:
                logger.info(
                    f"DEBUG: Class {defect} already exists for today for user {init_user_id} and location_id={location_id}")
                return True

        # Handle user-specific records for normal (non-initialization) updates
        if user_id:
            # Look for a record from TODAY with matching user_id, defect, and location_id
            if location_id:
                user_check_query = """
                SELECT structural_id, count, id, timestamp, location_id FROM defects 
                WHERE defect = %s AND id = %s AND location_id = %s AND timestamp BETWEEN %s AND %s
                """
                cursor.execute(user_check_query, (defect, user_id,
                               location_id, today_start, today_end))
            else:
                user_check_query = """
                SELECT structural_id, count, id, timestamp, location_id FROM defects 
                WHERE defect = %s AND id = %s AND timestamp BETWEEN %s AND %s
                """
                cursor.execute(user_check_query, (defect, user_id, today_start, today_end))

            user_specific_record = cursor.fetchone()

            if user_specific_record:
                # Update the user-specific record for today
                structural_id, existing_count, record_user_id, timestamp, existing_location_id = user_specific_record if len(
                    user_specific_record) >= 5 else (*user_specific_record, None)

                new_count = existing_count + count
                logger.info(f"DEBUG: Found today's record for {defect}: structural_id={structural_id}, "
                            f"existing_count={existing_count}, user_id={record_user_id}, "
                            f"timestamp={timestamp}, location_id={existing_location_id}")
                logger.info(
                    f"DEBUG: Will update today's count from {existing_count} to {new_count} for user {user_id}")

                # Update the record
                update_query = """
                UPDATE defects SET count = %s, threshold = %s WHERE structural_id = %s
                """
                cursor.execute(update_query, (new_count, threshold, structural_id))

                logger.info(
                    f"DEBUG: Updated today's record for {defect} with count={new_count}, user_id={user_id}")

                connection.commit()
                logger.info(f"DEBUG: Database commit successful for defect: {defect}")
                return True
            else:
                # No existing record for today with this location - create a new one
                logger.info(
                    f"DEBUG: No record found for today for {defect}, user {user_id}, and location_id={location_id}, creating new record with count={count}")

                # Include location_id in the insert if available
                if location_id:
                    insert_query = """
                    INSERT INTO defects (id, defect, count, threshold, timestamp, location_id) 
                    VALUES (%s, %s, %s, %s, %s, %s)
                    """
                    cursor.execute(insert_query, (user_id, defect, count,
                                   threshold, current_timestamp, location_id))
                    logger.info(f"DEBUG: Included location_id={location_id} in new record")
                else:
                    insert_query = """
                    INSERT INTO defects (id, defect, count, threshold, timestamp) 
                    VALUES (%s, %s, %s, %s, %s)
                    """
                    cursor.execute(insert_query, (user_id, defect,
                                   count, threshold, current_timestamp))

                logger.info(
                    f"DEBUG: New record inserted for {defect} with count={count}, user_id={user_id}, timestamp={current_timestamp}")

                connection.commit()
                logger.info(f"DEBUG: Database commit successful for defect: {defect}")
                return True

        # If we don't have a user ID, fall back to admin user with the same date-based logic
        admin_id = "A001"  # Default to admin for records without a specific user

        # If no location_id specified, try to get location_id for admin
        if location_id is None:
            location_id = get_latest_location_id(admin_id)
            logger.info(f"DEBUG: Retrieved location_id={location_id} for admin user as fallback")

        # Look for a record from TODAY with admin ID and matching location_id
        if location_id:
            check_query = """
            SELECT structural_id, count, id, timestamp, location_id FROM defects 
            WHERE defect = %s AND id = %s AND location_id = %s AND timestamp BETWEEN %s AND %s
            """
            cursor.execute(check_query, (defect, admin_id, location_id, today_start, today_end))
        else:
            check_query = """
            SELECT structural_id, count, id, timestamp, location_id FROM defects 
            WHERE defect = %s AND id = %s AND timestamp BETWEEN %s AND %s
            """
            cursor.execute(check_query, (defect, admin_id, today_start, today_end))

        admin_record = cursor.fetchone()

        if admin_record:
            # Update admin's existing record for today
            structural_id, existing_count, record_user_id, timestamp, existing_location_id = admin_record if len(
                admin_record) >= 5 else (*admin_record, None)

            new_count = existing_count + count
            logger.info(f"DEBUG: Found today's admin record for {defect}: structural_id={structural_id}, "
                        f"existing_count={existing_count}, user_id={record_user_id}, timestamp={timestamp}, location_id={existing_location_id}")
            logger.info(f"DEBUG: Will update today's count from {existing_count} to {new_count}")

            update_query = """
            UPDATE defects SET count = %s, threshold = %s WHERE structural_id = %s
            """
            cursor.execute(update_query, (new_count, threshold, structural_id))

            logger.info(f"DEBUG: Updated today's admin record for {defect} with count={new_count}")
        else:
            # No existing record for today with this location - create a new one for admin
            logger.info(
                f"DEBUG: No record found for today for {defect}, admin, and location_id={location_id}, creating new record with count={count}")

            # Include location_id in the insert if available
            if location_id:
                insert_query = """
                INSERT INTO defects (id, defect, count, threshold, timestamp, location_id) 
                VALUES (%s, %s, %s, %s, %s, %s)
                """
                cursor.execute(insert_query, (admin_id, defect, count,
                               threshold, current_timestamp, location_id))
                logger.info(f"DEBUG: Included location_id={location_id} in new admin record")
            else:
                insert_query = """
                INSERT INTO defects (id, defect, count, threshold, timestamp) 
                VALUES (%s, %s, %s, %s, %s)
                """
                cursor.execute(insert_query, (admin_id, defect,
                               count, threshold, current_timestamp))

            logger.info(
                f"DEBUG: New admin record inserted for {defect} with count={count}, user_id={admin_id}, timestamp={current_timestamp}")

        connection.commit()
        logger.info(f"DEBUG: Database commit successful for defect: {defect}")
        return True

    except Exception as e:
        logger.error(f"DEBUG: Error updating defect count: {e}")
        logger.error(f"DEBUG: Traceback: {traceback.format_exc()}")
        connection.rollback()
        return False

    finally:
        if connection.is_connected():
            cursor.close()
            connection.close()
            logger.info("DEBUG: Database connection closed")


def get_latest_location_id(user_id):
    """
    Get the most recent location_id for a user from the location table.

    Args:
        user_id (str): The user ID to get the location for

    Returns:
        int or None: The most recent location_id for the user, or None if not found
    """
    connection = get_connection()
    if not connection:
        logger.error(f"Cannot get latest location_id: No database connection")
        return None

    try:
        cursor = connection.cursor(dictionary=True)

        # Get the most recent location record for this user
        query = """
        SELECT location_id FROM location 
        WHERE id = %s 
        ORDER BY timestamp DESC 
        LIMIT 1
        """
        cursor.execute(query, (user_id,))
        result = cursor.fetchone()

        if result:
            location_id = result['location_id']
            logger.info(f"Found latest location_id {location_id} for user {user_id}")
            return location_id
        else:
            logger.warning(f"No location record found for user {user_id}")
            return None

    except Exception as e:
        logger.error(f"Error getting latest location_id: {e}")
        # Provide more details about the error for debugging
        import traceback
        logger.debug(f"Traceback: {traceback.format_exc()}")
        return None
    finally:
        if connection.is_connected():
            cursor.close()
            connection.close()


def store_management_access(user_identifier, mgmt_username, password, name=None, phone_no=None, mail=None):
    """
    Store management access credentials in the management table.
    This function is called when a user grants access via the Grant Access dialog.
    Checks for username uniqueness in both management and crm tables.

    Args:
        user_identifier (str): Either the user_id or username of the logged-in user granting access
        mgmt_username (str): The username for management access
        password (str): The password for management access
        name (str, optional): The full name for management access
        phone_no (str, optional): The phone number for management access
        mail (str, optional): The email address for management access

    Returns:
        tuple: (status_code, message)
            status_code: 0 for success, 1 for username already exists, -1 for error
            message: Description of the result
    """
    connection = get_connection()
    if not connection:
        logger.error("Cannot store management access data: No database connection")
        return (-1, "Database connection failed")

    try:
        cursor = connection.cursor()

        # Generate a unique m_id
        m_id = f"M{uuid.uuid4().hex[:6].upper()}"

        # First, determine if user_identifier is a user_id or username
        user_id = None
        # Check if user_identifier looks like a user ID (usually starts with letters followed by numbers)
        if user_identifier and (user_identifier.startswith('SR') or user_identifier.startswith('A')):
            # It's likely a user ID, so use it directly
            user_id = user_identifier
            logger.info(f"Using directly provided user_id: {user_id}")
        elif user_identifier:
            # Treat it as a username and look up the ID
            user_id = get_user_id_by_username(user_identifier)
            if not user_id:
                logger.warning(f"Could not find user ID for username '{user_identifier}', using fallback")
                # Attempt to load from session state as fallback
                try:
                    session_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "session_state.json")
                    if os.path.exists(session_file):
                        with open(session_file, 'r') as f:
                            session_data = json.load(f)
                            if 'user_info' in session_data and 'user_id' in session_data['user_info']:
                                user_id = session_data['user_info']['user_id']
                                logger.info(f"Retrieved user_id from session: {user_id}")
                            elif 'current_user_id' in session_data:
                                user_id = session_data['current_user_id']
                                logger.info(f"Retrieved user_id from session current_user_id: {user_id}")
                except Exception as e:
                    logger.error(f"Error retrieving user ID from session: {e}")
            
            # If still no user_id, use a default only as last resort
            if not user_id:
                user_id = "A001"  # Default admin user as last resort
                logger.warning(f"No user_id found, using default admin ID: {user_id}")
        else:
            logger.warning("No user identifier provided for management access")
            user_id = "A001"  # Default to admin if no username provided

        # First check if username already exists in management table
        # Use case-sensitive comparison to allow similar usernames with different cases
        check_mgmt_query = "SELECT m_id FROM management WHERE username = %s"
        cursor.execute(check_mgmt_query, (mgmt_username,))
        existing_mgmt_record = cursor.fetchone()

        if existing_mgmt_record:
            # Return username exists status instead of updating
            logger.info(f"Username already exists in management table: {mgmt_username}")
            return (1, f"Username '{mgmt_username}' already exists in management")

        # Now check if username exists in crm table
        check_crm_query = "SELECT crm_id FROM crm WHERE cusername = %s"
        cursor.execute(check_crm_query, (mgmt_username,))
        existing_crm_record = cursor.fetchone()

        if existing_crm_record:
            # Return username exists status 
            logger.info(f"Username already exists in crm table: {mgmt_username}")
            return (1, f"Username '{mgmt_username}' already exists in CRM system")

        # If we get here, the username is unique in both tables
        # Insert a new record with all fields - user_id is used here, not username
        insert_query = """
        INSERT INTO management (m_id, id, username, password, name, phone_no, mail) 
        VALUES (%s, %s, %s, %s, %s, %s, %s)
        """
        cursor.execute(insert_query, (m_id, user_id, mgmt_username,
                       password, name, phone_no, mail))
        connection.commit()

        # Log using the user_id instead of username for clarity
        logger.info(
            f"Stored management access with m_id {m_id} for user_id {user_id}, management username: {mgmt_username}")
        return (0, "Access granted successfully")

    except Exception as e:
        logger.error(f"Error storing management access data: {e}")
        connection.rollback()
        return (-1, f"Database error: {str(e)}")
    finally:
        if connection.is_connected():
            cursor.close()
            connection.close()


def store_total_defect_count(user_id=None, total_count=0, location_id=None):
    """
    Store the total defect count in the total_defect table.
    This function records the total number of rejected parts.
    Updates existing records from the same day rather than creating new ones.

    Args:
        user_id (str, optional): The user ID to associate with this record.
        total_count (int): The total count of defects/rejected parts.
        location_id (int, optional): The location ID to associate with this record.

    Returns:
        bool: True if data was stored successfully, False otherwise
    """
    logger.info(
        f"DEBUG: store_total_defect_count called with user_id={user_id}, "
        f"total_count={total_count}, location_id={location_id}")

    connection = get_connection()
    if not connection:
        logger.error("Cannot store total defect count: No database connection")
        return False

    try:
        cursor = connection.cursor(buffered=True)

        # Get the current date in YYYY-MM-DD format
        current_date = datetime.now().strftime('%Y-%m-%d')
        
        # For database queries: start and end of the current day
        today_start = f"{current_date} 00:00:00"
        today_end = f"{current_date} 23:59:59"
        
        # Generate the current timestamp
        current_timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        # If user_id is not provided or is None/empty, try to get it from session state
        if not user_id:
            try:
                # Try to load the current user from session state
                session_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "session_state.json")
                if os.path.exists(session_file):
                    with open(session_file, 'r') as f:
                        session_data = json.load(f)
                        if 'user_info' in session_data and session_data['user_info']:
                            user_info = session_data['user_info']
                            user_id = user_info.get('user_id') or user_info.get('id')
                            logger.info(f"Retrieved user_id from session: {user_id}")
                        else:
                            logger.warning("No user_info found in session_state.json")
                else:
                    logger.warning("session_state.json file not found")
                    
                # If still no user_id, default to admin as last resort
                if not user_id:
                    user_id = "A001"  # Default admin user
                    logger.warning(f"No user_id found in session, using default admin ID: {user_id}")
            except Exception as session_error:
                logger.error(f"Error reading session state: {session_error}")
                user_id = "A001"  # Default admin user
                logger.warning(f"Session read failed, using default admin ID: {user_id}")

        # If location_id is not provided, try to fetch it
        if location_id is None and user_id:
            location_id = get_latest_location_id(user_id)
            logger.info(f"Retrieved location_id={location_id} for user_id={user_id}")

        # Check if a record already exists for today with this user_id and location_id
        if location_id:
            check_query = """
            SELECT td_id, tcount FROM total_defect 
            WHERE id = %s AND location_id = %s AND timestamp BETWEEN %s AND %s
            """
            cursor.execute(check_query, (user_id, location_id, today_start, today_end))
        else:
            check_query = """
            SELECT td_id, tcount FROM total_defect 
            WHERE id = %s AND timestamp BETWEEN %s AND %s
            """
            cursor.execute(check_query, (user_id, today_start, today_end))

        existing_record = cursor.fetchone()

        if existing_record:
            # Update existing record instead of creating a new one
            td_id, existing_count = existing_record
            
            # Only update if the new count is different
            if total_count != existing_count:
                update_query = """
                UPDATE total_defect SET tcount = %s, timestamp = %s WHERE td_id = %s
                """
                cursor.execute(update_query, (total_count, current_timestamp, td_id))
                connection.commit()
                logger.info(
                    f"Updated total defect count from {existing_count} to {total_count} "
                    f"for user {user_id}, td_id={td_id}, location_id={location_id}")
            else:
                logger.info(
                    f"Total count unchanged ({total_count}), no update needed "
                    f"for user {user_id}, td_id={td_id}, location_id={location_id}")
            
            return True
        else:
            # No record for today - create a new one
            insert_query = """
            INSERT INTO total_defect (id, timestamp, tcount, location_id) 
            VALUES (%s, %s, %s, %s)
            """
            cursor.execute(insert_query, (user_id, current_timestamp, total_count, location_id))
            connection.commit()

            logger.info(
                f"Created new total defect count record: {total_count} for user {user_id}, "
                f"timestamp={current_timestamp}, location_id={location_id}")
            
            # Log success message to match expected format from logs
            logger.info(f"Saved total rejected count ({total_count}) to database for user {user_id}")
            return True

    except Exception as e:
        logger.error(f"Error storing total defect count: {e}")
        # Provide more details about the error for debugging
        import traceback
        logger.debug(f"Traceback: {traceback.format_exc()}")
        connection.rollback()
        return False
    finally:
        if connection.is_connected():
            cursor.close()
            connection.close()
            logger.info("Database connection closed after storing total defect count")


# For testing
if __name__ == "__main__":
    # Test connection
    if test_connection():
        print("Successfully connected to Azure MySQL database")
    else:
        print("Failed to connect to Azure MySQL database")
