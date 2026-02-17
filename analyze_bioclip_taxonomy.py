from bioclip import TreeOfLifeClassifier
import pandas as pd

def analyze_counts():
    print("Loading BioCLIP label data...")
    cls = TreeOfLifeClassifier()
    df = cls.get_label_data()
    
    print(f"Total families: {len(df['family'].unique())}")
    
    # Count families per Phylum
    phyla_counts = df.groupby('phylum')['family'].nunique().sort_values(ascending=False)
    print("\nTop 20 Phyla by Family count:")
    print(phyla_counts.head(20))
    
    # Count families per Class for Arthropoda/Chordata
    print("\n--- Arthropoda Classes ---")
    arthCheck = df[df['phylum'] == 'Arthropoda']
    print(arthCheck.groupby('class')['family'].nunique().sort_values(ascending=False).head(10))
    
    print("\n--- Chordata Classes ---")
    chordCheck = df[df['phylum'] == 'Chordata']
    print(chordCheck.groupby('class')['family'].nunique().sort_values(ascending=False).head(10))
    
    print("\n--- Tracheophyta (Plants) Classes ---")
    plantCheck = df[df['phylum'] == 'Tracheophyta']
    print(plantCheck.groupby('class')['family'].nunique().sort_values(ascending=False).head(5))

if __name__ == "__main__":
    analyze_counts()
