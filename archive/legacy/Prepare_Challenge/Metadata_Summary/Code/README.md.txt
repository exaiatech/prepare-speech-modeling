# Metadata Analysis and Visualization Script

### Purpose
This script is designed for Exploratory Data Analysis (EDA) of the PREPARE Challenge dataset. Before running the main ASR and NLP pipelines, this module processes the dataset's metadata to generate clear, professional visualizations. It provides immediate insights into the distribution of patient demographics (Age, Gender), clinical diagnoses (Control, MCI, AD, PPA), and recording languages, ensuring the research team understands the data balance.

### Inputs
* **Required Files:** The aggregated metadata summary file.
* **File Format:** `.csv` file (e.g., `metadata_summary.csv`).
* **Parameters:** The script requires the CSV to contain `Category`, `Value`, and `Count` columns to accurately map the distributions.

### Outputs
* **Generated Outputs:** High-quality, publication-ready visualization charts.
* **File Format:** `.png` image files.
* **Structure:** Saves individual charts for each primary category (e.g., `Gender_distribution.png`, `Diagnosis_distribution.png`), alongside specialized age distribution charts (grouped pie chart and exact age bar chart). All outputs are saved to the designated `OUTPUT_DIR`.

### Configuration
The script utilizes hardcoded constant variables at the top for easy path management:
* `INPUT_FILE`: The path or filename of the summary dataset (Default: `metadata_summary.csv`).
* `OUTPUT_DIR`: The destination folder where all generated `.png` charts will be saved. The script automatically creates this directory if it does not exist.

### Dependencies
The following Python libraries are required to run this script:
* `pandas` (for data manipulation and reading the CSV)
* `matplotlib.pyplot` (for base plotting mechanics)
* `seaborn` (for advanced, aesthetically pleasing statistical graphics)
* Built-in Python modules: `os`

### Execution Instructions
1. Ensure the required visualization libraries are installed:
   ```bash
   pip install pandas matplotlib seaborn