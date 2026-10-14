import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# --- CONFIGURATION ---
INPUT_FILE = r"C:\Users\arday\Desktop\UROP\Project_Codes\Prepare_Challenge\Metadata_Summary\Output\metadata_summary.csv"
# Change this path to where you want to save the images
OUTPUT_DIR = r"C:\Users\arday\Desktop\UROP\Project_Codes\Prepare_Challenge\Metadata_Visualizations\Output" 


def visualize_data():
    """
    Reads the metadata summary and saves visualizations to the chosen output directory.
    """
    # Create the output folder if it does not exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. Load Data
    df = pd.read_csv(INPUT_FILE)
    
    # 2. Basic Categories (Gender, Split, Diagnosis, Language)
    categories = ['Gender', 'Split', 'Diagnosis', 'Language']
    
    for cat in categories:
        sub_df = df[df['Category'] == cat].copy()
        sub_df['Count'] = pd.to_numeric(sub_df['Count'])
        sub_df = sub_df.sort_values('Count', ascending=False)
        
        plt.figure(figsize=(8, 5))
        sns.barplot(x='Value', y='Count', data=sub_df, palette='viridis')
        plt.title(f'Distribution of {cat}')
        plt.ylabel('Count')
        plt.xlabel(cat)
        plt.tight_layout()
        
        # Save to output directory
        save_path = os.path.join(OUTPUT_DIR, f'{cat}_distribution.png')
        plt.savefig(save_path)
        plt.close()

    # 3. Advanced Age Processing
    age_df = df[df['Category'] == 'Age'].copy()
    age_df['Value'] = pd.to_numeric(age_df['Value'], errors='coerce')
    age_df['Count'] = pd.to_numeric(age_df['Count'])
    age_df = age_df.dropna(subset=['Value'])

    # --- AGE CHART A: Grouped Pie Chart ---
    bins = [0, 49, 59, 69, 79, 89, 150]
    labels = ['<50', '50-59', '60-69', '70-79', '80-89', '90+']
    age_df['Age_Group'] = pd.cut(age_df['Value'], bins=bins, labels=labels, right=True)
    
    age_groups = age_df.groupby('Age_Group', observed=True)['Count'].sum().reset_index()
    age_groups = age_groups[age_groups['Count'] > 0] 

    plt.figure(figsize=(8, 8))
    plt.pie(age_groups['Count'], labels=age_groups['Age_Group'], autopct='%1.1f%%', startangle=140, colors=sns.color_palette('pastel'))
    plt.title('Age Distribution (Grouped)')
    
    # Save to output directory
    save_path_pie = os.path.join(OUTPUT_DIR, 'Age_grouped_pie.png')
    plt.savefig(save_path_pie)
    plt.close()

    # --- AGE CHART B: Exact Age Bar Chart ---
    age_df_sorted = age_df.sort_values(by='Value').reset_index(drop=True)
    plt.figure(figsize=(15, 6))
    
    max_count = age_df_sorted['Count'].max()
    min_count = age_df_sorted['Count'].min()
    
    colors = []
    for count in age_df_sorted['Count']:
        if count == max_count:
            colors.append('red') 
        elif count == min_count:
            colors.append('blue') 
        else:
            colors.append('lightgray')

    age_df_sorted['Value'] = age_df_sorted['Value'].astype(int).astype(str)
    
    plt.bar(age_df_sorted['Value'], age_df_sorted['Count'], color=colors)
    plt.title('Exact Age Distribution (Red: Most Frequent, Blue: Least Frequent)')
    plt.xlabel('Age')
    plt.ylabel('Count')
    plt.xticks(rotation=90, fontsize=8)
    plt.tight_layout()
    
    # Save to output directory
    save_path_bar = os.path.join(OUTPUT_DIR, 'Age_exact_distribution.png')
    plt.savefig(save_path_bar)
    plt.close()

    print(f"All charts generated and saved successfully to:\n{OUTPUT_DIR}")

if __name__ == "__main__":
    visualize_data()