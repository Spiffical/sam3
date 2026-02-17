import pyworms
import pandas as pd
import numpy as np
import time
import json
import os
from bioclip import TreeOfLifeClassifier

def build_worms_filter_list(output_path: str):
    print("Loading BioCLIP label data...")
    cls = TreeOfLifeClassifier()
    df = cls.get_label_data()
    
    # Get unique families with taxonomy
    fam_tax = df[['family', 'phylum', 'class']].drop_duplicates('family')
    
    print(f"Total BioCLIP Families: {len(fam_tax)}")
    
    # 1. Defined Heuristics to speed up API process
    # We accept these without query because they are definitionally 99% marine/aquatic 
    # and we want to include them. (Exceptions like freshwater sponges exist but are rare distractions).
    SAFE_MARINE_PHYLA = {
        "Porifera", "Ctenophora", "Cnidaria", "Echinodermata", 
        "Brachiopoda", "Bryozoa", "Nemertea", "Chaetognatha",
        "Hemichordata", "Xenacoelomorpha", "Rhodophyta", "Ochrophyta"
    }
    
    # Accept these Classes immediately
    SAFE_MARINE_CLASSES = {
        "Polychaeta", "Cephalopoda", "Scaphopoda",
        "Ascidiacea", "Thaliacea", "Appendicularia",
        "Pycnogonida", "Cirripedia", "Thecostraca"
    }
    
    # Reject these immediately (Terrestrial/Non-marine focus)
    EXCLUDED_PHYLA = {
        "Tracheophyta", "Basidiomycota", "Ascomycota", "Bryophyta", 
        "Magnoliophyta" # Plants/Fungi
    }
    
    # Reject these Classes
    EXCLUDED_CLASSES = {
        "Insecta", "Arachnida", "Diplopoda", "Chilopoda", "Collembola",
        "Aves", "Amphibia", "Reptilia", "Mammalia" 
        # Note: Marine mammals exist, but usually distinct families. 
        # For this optimization we might skip them or query them.
        # Let's QUERY "Mammalia" just in case of whales/seals? 
        # "Reptilia" (Turtles). 
        # So remove them from Exclude for now to be safe, or just exclude to save time if user focuses on invetebreates.
        # User said "underwater things ONLY".
        # Let's keep Insecta/Arachnida/Plants as strict excludes (huge savings).
    }
    
    candidates_to_query = []
    valid_worms_families = []
    
    for idx, row in fam_tax.iterrows():
        fam = row['family']
        ph = row['phylum']
        cl = row['class']
        
        if not isinstance(fam, str) or not fam: continue
        
        # Heuristic Logic
        if ph in SAFE_MARINE_PHYLA:
            valid_worms_families.append(fam)
        elif cl in SAFE_MARINE_CLASSES:
            valid_worms_families.append(fam)
        elif ph in EXCLUDED_PHYLA:
            continue
        elif cl in EXCLUDED_CLASSES:
            continue
        elif cl in ["Mammalia", "Reptilia", "Aves"]:
             # Optional: Skip for speed unless requested?
             # Let's skip Aves (Birds). Mammalia/Reptilia might have sea turtles/whales.
             # Check Mammals?
             candidates_to_query.append(fam)
        else:
            # Everything else (Mollusca, Fish, Crustacea, etc) -> Query
            candidates_to_query.append(fam)
            
    print(f"Heuristics: {len(valid_worms_families)} auto-accepted. {len(candidates_to_query)} to query.")
    
    # 2. Batch Query
    candidates_to_query = candidates_to_query
    
    batch_size = 50 
    print("Starting WoRMS verification for candidates...")
    start_time = time.time()
    total = len(candidates_to_query)
    
    for i in range(0, total, batch_size):
        chunk = candidates_to_query[i:i+batch_size]
        curr_valid = []
        
        try:
            resp = pyworms.aphiaRecordsByMatchNames(chunk)
            
            for name, records in zip(chunk, resp):
                if records:
                    rec = records[0] 
                    im = rec.get('isMarine') in (1, True, '1')
                    ib = rec.get('isBrackish') in (1, True, '1')
                    if im or ib:
                        curr_valid.append(name)
                        
            valid_worms_families.extend(curr_valid)
            
        except Exception as e:
            print(f"Error checking batch {i}: {e}")
            
        if i % 100 == 0:
             elapsed = time.time() - start_time
             if elapsed > 0:
                 rate = (i+batch_size) / elapsed
                 remaining = (total - i) / rate
                 print(f"Processed {i}/{total}. Valid Found: {len(curr_valid)}. Total Valid: {len(valid_worms_families)}. ETA: {remaining/60:.1f} min")

    print(f"Finished. Found {len(valid_worms_families)} valid marine families total.")
    
    with open(output_path, 'w') as f:
        json.dump(valid_worms_families, f)
        
    print(f"Saved filter list to {output_path}")

if __name__ == "__main__":
    out = "assets/taxonomy/WoRMS_marine_families.json"
    if not os.path.exists(os.path.dirname(out)):
        os.makedirs(os.path.dirname(out))
    build_worms_filter_list(out)
