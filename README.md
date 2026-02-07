# Data Setup Instructions

## Installing Requirements

First, install all required packages:

```bash
pip install -r requirements.txt
```

## Data Path Configuration

**Important:** The dataset is not included in the repository due to its size (7GB). Since each team member stores the data in a different location, we need to configure the path individually to ensure the project runs correctly during grading.

### Setup Steps

1. Create a `.env` file in the project root directory
2. Add your local data path to the file

### Example

If your data is located at `/home/bagga/Desktop/pos_ud_project/data/raw/allzip/ud-treebanks-v2.17/UD_English-EWT`, your `.env` file should contain:

```
UD_DATA_PATH=/home/bagga/Desktop/pos_ud_project/data/raw/allzip/ud-treebanks-v2.17/UD_English-EWT
```

### For the Instructor

When testing this project, please:
1. Download the UD Treebanks v2.17 dataset from: 
`https://lindat.mff.cuni.cz/repository/items/b4fcb1e0-f4b2-4939-80f5-baeafda9e5c0`
2. Extract it to your preferred location
3. Create a `.env` file in the project root
4. Set `UD_DATA_PATH` to point to your `UD_English-EWT` directory

This approach ensures path compatibility across different systems without hardcoding absolute paths in the source code.