# Role Mining Project

## Overview

The Role Mining Project aims to analyze user permissions and roles within an organization. This project includes a script for fetching user permissions from an API (`api.py`) and another script for analyzing and visualizing these permissions (`main.py`). The main goal is to optimize role assignments to maximize coverage of permissions.

## Features

- Fetches user permissions from a remote API.
- Analyzes permissions to find the most similar roles.
- Visualizes the results with heatmaps and other charts.
- Saves the results in various formats including CSV, Excel, and image files.

## Requirements

- Python 3.x
- Required libraries: `requests`, `requests_negotiate_sspi`, `pandas`, `numpy`, `matplotlib`, `seaborn`, `openpyxl`

You can install the required libraries using pip:

```bash
pip install requests requests_negotiate_sspi pandas numpy matplotlib seaborn openpyxl
```

## Scripts

### `api.py`

This script handles the fetching of user permissions from a remote API. The main function is `fetch_user_permissions`, which:

1. Takes a group name and a save path as inputs.
2. Fetches the list of members in the specified group.
3. For each member, retrieves their assigned roles.
4. Saves the results into CSV files and binary pickle files for further analysis.

**Usage:**

```python
from api import fetch_user_permissions

fetch_user_permissions(save_path, group_name)
```

**Parameters:**

- `save_path`: The directory where the results will be saved.
- `group_name`: The name of the group to fetch permissions for.

### `main.py`

This script is responsible for analyzing the permissions data and visualizing it. It performs the following tasks:

1. Reads the permission and user data from CSV and pickle files.
2. Computes similarities between roles based on user permissions.
3. Updates the role and permission matrices to merge similar roles.
4. Calculates and prints the coverage of permissions by roles.
5. Visualizes the results in a heatmap and saves it as an image.
6. Saves the results in an Excel file and CSV files.

**Usage:**

```python
python main.py
```

**Interactive Inputs:**

- Enter the group name for fetching permissions.
- Enter the maximum number of roles for optimization.
- Provide the folder path where results will be saved.

## File Outputs

- `user_permissions.csv`: CSV file containing user permissions.
- `data_permissions.csv`: CSV file listing all unique permissions.
- `list_of_users.txt`: Pickle file containing the list of users.
- `list_of_permissions.txt`: Pickle file containing the list of permissions.
- `users_to_roles.csv`: CSV file mapping users to roles.
- `heatmap1.png`: Heatmap image visualizing permissions coverage.
- `matrix.xlsx`: Excel file with the binary matrix of permissions vs users.

## Example

1. Run `main.py` to start the analysis.
2. Input the group name when prompted.
3. Specify the maximum number of roles and folder path for saving results.
4. Review the output files for analysis and visualization.

## Contributing

Feel free to submit issues, feature requests, or pull requests. Contributions are welcome!
