import pandas as pd

# --- CONFIGURATION ---
INPUT_FILE = r"C:\Users\arday\Downloads\metadata.csv"  # Update this if your file has a different name
OUTPUT_FILE = r"C:\Users\arday\Desktop\UROP\metadata_summary.csv"


def analyze_dataset():
    """
    Reads the metadata, counts the values for specific categories, 
    prints the results, and saves them to a new CSV.
    """
    try:
        # 1. Load the dataset
        df = pd.read_csv(INPUT_FILE)
        print(f"Successfully loaded {INPUT_FILE}. Total rows: {len(df)}\n")
        
        # The 5 categories we want to analyze
        target_columns = ['age', 'gender', 'split', 'diagnosis', 'language']
        
        # This list will hold our final data to save as CSV
        summary_data = []

        # 2. Analyze each category
        for col in target_columns:
            print(f"--- {col.upper()} DISTRIBUTION ---")
            
            # value_counts() automatically counts how many times each item appears
            counts = df[col].value_counts(dropna=False)
            
            for value, count in counts.items():
                print(f"{value}: {count}")
                
                # Add the data to our list
                summary_data.append({
                    "Category": col.capitalize(),
                    "Value": value,
                    "Count": count
                })
            print("-" * 30)

        # 3. Save the results to a new CSV file
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(OUTPUT_FILE, index=False, encoding='utf-8')
        
        print(f"\n[SUCCESS] Summary report saved to: {OUTPUT_FILE}")

    except FileNotFoundError:
        print(f"[ERROR] Could not find {INPUT_FILE}. Please check the file name and path.")
    except Exception as e:
        print(f"[ERROR] An unexpected error occurred: {e}")


if __name__ == "__main__":
    analyze_dataset()