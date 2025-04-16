"""
File operations utilities for working with research data, documents, and code files.
"""
import os
import json
import yaml
import csv
from typing import Any, Dict, List, Optional, Union, Tuple
import re
from datetime import datetime


def ensure_directory(directory_path: str) -> bool:
    """
    Ensure a directory exists, creating it if necessary.
    
    Args:
        directory_path: Path to the directory
        
    Returns:
        True if directory exists or was created, False on failure
    """
    try:
        if not os.path.exists(directory_path):
            os.makedirs(directory_path, exist_ok=True)
        return os.path.isdir(directory_path)
    except Exception as e:
        print(f"Error creating directory {directory_path}: {str(e)}")
        return False

def safe_read_file(file_path: str, binary: bool = False) -> Tuple[bool, Union[str, bytes, None]]:
    """
    Safely read a file with error handling.
    
    Args:
        file_path: Path to the file
        binary: Whether to read in binary mode
        
    Returns:
        Tuple of (success, content)
    """
    try:
        mode = 'rb' if binary else 'r'
        with open(file_path, mode) as f:
            return True, f.read()
    except Exception as e:
        print(f"Error reading file {file_path}: {str(e)}")
        return False, None

def safe_write_file(file_path: str, content: Union[str, bytes], binary: bool = False) -> bool:
    """
    Safely write to a file with error handling.
    
    Args:
        file_path: Path to the file
        content: Content to write
        binary: Whether to write in binary mode
        
    Returns:
        True on success, False on failure
    """
    try:
        # Ensure directory exists
        directory = os.path.dirname(file_path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
            
        # Write the file
        mode = 'wb' if binary else 'w'
        with open(file_path, mode) as f:
            f.write(content)
        return True
    except Exception as e:
        print(f"Error writing file {file_path}: {str(e)}")
        return False

def load_json_file(file_path: str) -> Tuple[bool, Optional[Dict[str, Any]]]:
    """
    Load and parse a JSON file.
    
    Args:
        file_path: Path to the JSON file
        
    Returns:
        Tuple of (success, data)
    """
    success, content = safe_read_file(file_path)
    if not success or content is None:
        return False, None
        
    try:
        data = json.loads(content)
        return True, data
    except Exception as e:
        print(f"Error parsing JSON file {file_path}: {str(e)}")
        return False, None

def save_json_file(file_path: str, data: Any, indent: int = 2) -> bool:
    """
    Save data to a JSON file.
    
    Args:
        file_path: Path to save the file
        data: Data to serialize as JSON
        indent: Indentation level for pretty printing
        
    Returns:
        True on success, False on failure
    """
    try:
        json_str = json.dumps(data, indent=indent)
        return safe_write_file(file_path, json_str)
    except Exception as e:
        print(f"Error saving JSON file {file_path}: {str(e)}")
        return False

def load_yaml_file(file_path: str) -> Tuple[bool, Optional[Dict[str, Any]]]:
    """
    Load and parse a YAML file.
    
    Args:
        file_path: Path to the YAML file
        
    Returns:
        Tuple of (success, data)
    """
    success, content = safe_read_file(file_path)
    if not success or content is None:
        return False, None
        
    try:
        data = yaml.safe_load(content)
        return True, data
    except Exception as e:
        print(f"Error parsing YAML file {file_path}: {str(e)}")
        return False, None

def save_yaml_file(file_path: str, data: Any) -> bool:
    """
    Save data to a YAML file.
    
    Args:
        file_path: Path to save the file
        data: Data to serialize as YAML
        
    Returns:
        True on success, False on failure
    """
    try:
        yaml_str = yaml.dump(data, sort_keys=False)
        return safe_write_file(file_path, yaml_str)
    except Exception as e:
        print(f"Error saving YAML file {file_path}: {str(e)}")
        return False

def load_csv_file(file_path: str, has_header: bool = True) -> Tuple[bool, Optional[List[Dict[str, str]]]]:
    """
    Load and parse a CSV file.
    
    Args:
        file_path: Path to the CSV file
        has_header: Whether the CSV has a header row
        
    Returns:
        Tuple of (success, data)
    """
    try:
        with open(file_path, 'r', newline='') as f:
            if has_header:
                reader = csv.DictReader(f)
                return True, list(reader)
            else:
                reader = csv.reader(f)
                data = list(reader)
                return True, data
    except Exception as e:
        print(f"Error reading CSV file {file_path}: {str(e)}")
        return False, None

def save_csv_file(file_path: str, data: List[Dict[str, Any]], fieldnames: Optional[List[str]] = None) -> bool:
    """
    Save data to a CSV file.
    
    Args:
        file_path: Path to save the file
        data: List of dictionaries to save as CSV rows
        fieldnames: Optional list of field names (columns)
        
    Returns:
        True on success, False on failure
    """
    try:
        # If fieldnames not provided, get from first row
        if fieldnames is None and data:
            fieldnames = list(data[0].keys())
            
        with open(file_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(data)
        return True
    except Exception as e:
        print(f"Error saving CSV file {file_path}: {str(e)}")
        return False

def create_timestamped_file(base_path: str, extension: str = "txt") -> str:
    """
    Create a filename with a timestamp to avoid overwrites.
    
    Args:
        base_path: Base path and filename
        extension: File extension (without the dot)
        
    Returns:
        Path with timestamp added
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Split the path into directory, basename, and extension
    directory = os.path.dirname(base_path)
    basename = os.path.basename(base_path)
    
    # Strip any existing extension from basename
    basename = os.path.splitext(basename)[0]
    
    # Create the new filename with timestamp
    filename = f"{basename}_{timestamp}.{extension}"
    
    # Join with directory if it exists
    if directory:
        return os.path.join(directory, filename)
    return filename

def search_files(directory: str, pattern: str, recursive: bool = True) -> List[str]:
    """
    Search for files matching a pattern in a directory.
    
    Args:
        directory: Directory to search
        pattern: Regex pattern for matching file names
        recursive: Whether to search subdirectories
        
    Returns:
        List of matching file paths
    """
    matches = []
    try:
        for root, dirs, files in os.walk(directory):
            for filename in files:
                if re.match(pattern, filename):
                    matches.append(os.path.join(root, filename))
            
            # Don't recurse into subdirectories if not requested
            if not recursive:
                break
                
        return matches
    except Exception as e:
        print(f"Error searching files: {str(e)}")
        return []

def get_file_info(file_path: str) -> Dict[str, Any]:
    """
    Get information about a file.
    
    Args:
        file_path: Path to the file
        
    Returns:
        Dictionary with file information
    """
    try:
        if not os.path.exists(file_path):
            return {"exists": False}
            
        stat_info = os.stat(file_path)
        return {
            "exists": True,
            "path": file_path,
            "name": os.path.basename(file_path),
            "directory": os.path.dirname(file_path),
            "size": stat_info.st_size,
            "created": datetime.fromtimestamp(stat_info.st_ctime).isoformat(),
            "modified": datetime.fromtimestamp(stat_info.st_mtime).isoformat(),
            "is_file": os.path.isfile(file_path),
            "is_dir": os.path.isdir(file_path),
            "extension": os.path.splitext(file_path)[1]
        }
    except Exception as e:
        return {
            "exists": os.path.exists(file_path),
            "error": str(e)
        }
        
