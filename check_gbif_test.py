
import requests
import json
import time

def check_gbif(species_name, country_code='CA'):
    print(f"\n--- Checking GBIF for {species_name} in {country_code} ---")
    
    # 1. Match Species Name
    try:
        match_url = "https://api.gbif.org/v1/species/match"
        params = {'name': species_name, 'kingdom': 'Animalia'}
        r = requests.get(match_url, params=params)
        data = r.json()
        
        usage_key = data.get('usageKey')
        if not usage_key:
            print("  Species not found in GBIF backbone.")
            return
            
        accepted_name = data.get('canonicalName')
        print(f"  Matched: {accepted_name} (Key: {usage_key})")
        
        # 2. Check Occurrences in Country
        occ_url = "https://api.gbif.org/v1/occurrence/search"
        occ_params = {
            'taxonKey': usage_key,
            'country': country_code,
            'limit': 0 # We just want the count
        }
        r_occ = requests.get(occ_url, params=occ_params)
        occ_data = r_occ.json()
        count = occ_data.get('count', 0)
        
        print(f"  Occurrences in {country_code}: {count}")
        
        # 3. Check Global Occurrences to verify commonness
        occ_params_global = {
            'taxonKey': usage_key,
            'limit': 0
        }
        r_glob = requests.get(occ_url, params=occ_params_global)
        global_count = r_glob.json().get('count', 0)
        print(f"  Global Occurrences: {global_count}")
        
        return count > 0
        
    except Exception as e:
        print(f"  GBIF API Error: {e}")
        return False

if __name__ == "__main__":
    # Test Cases
    check_gbif("Oxyeleotris marmorata") # Marble Goby (Should be 0 in Canada)
    check_gbif("Doclea brachyrhynchos") # Spider Crab (Should be 0?)
    check_gbif("Grimothea planipes")    # Squat Lobster (Should be > 0)
    check_gbif("Gadus morhua")          # Atlantic Cod (Should be > 0 in CA - Atlantic side)
    check_gbif("Pandalus platyceros")   # Spot Prawn (Should be > 0 in CA)
