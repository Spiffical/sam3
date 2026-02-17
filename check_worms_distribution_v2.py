
import pyworms

def check_dist(name):
    print(f"--- Checking Distributions for {name} ---")
    res = pyworms.aphiaRecordsByMatchNames([name])
    if res and res[0]:
        record = res[0][0]
        aphia_id = record['AphiaID']
        print(f"AphiaID: {aphia_id}")
        
        try:
            dists = pyworms.aphiaDistributionsByAphiaID(aphia_id)
            print(f"Found {len(dists)} distribution records.")
            if dists:
                # Print first few to see structure
                for i, d in enumerate(dists[:3]):
                    print(f"Dist {i}: {d}")
                
                # Check for region names
                regions = set([d.get('locality') for d in dists if d.get('locality')])
                print("Localities:", list(regions)[:5])
                
                areas = set([d.get('establishmentMeans') for d in dists]) # maybe not relevant
                
                # Is there a standardized region code like MRGID?
                # Usually WoRMS links to MarineRegions.org
                
        except Exception as e:
            print(f"Error fetching distributions: {e}")

if __name__ == "__main__":
    check_dist("Oxyeleotris marmorata")
