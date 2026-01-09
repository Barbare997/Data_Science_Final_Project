"""
Path setup for Jupyter notebooks.

Helps notebooks find the src directory.
"""

import sys
from pathlib import Path


def setup_project_path():
    """
    Add project root to Python path so notebooks can import from src.
    """
    current_dir = Path().resolve()
    
    # Check if we're in notebooks directory or project root
    if (current_dir / 'src').exists():
        # We're in project root
        project_root = current_dir
    elif (current_dir.parent / 'src').exists():
        # We're in notebooks directory, go up one level
        project_root = current_dir.parent
    else:
        # Try to find project root by looking for src directory
        project_root = current_dir
        while project_root != project_root.parent:
            if (project_root / 'src').exists():
                break
            project_root = project_root.parent
    
    # Add project root to Python path
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    
    return project_root


if __name__ == '__main__':
    root = setup_project_path()
    print(f"✓ Project root added to path: {root}")
    print(f"✓ You can now import from src: from src.data_processing import ...")

