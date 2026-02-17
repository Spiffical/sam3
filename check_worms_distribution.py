
import pyworms

def check_species(name):
    print(f"--- Checking {name} ---")
    # Get AphiaID first
    res = pyworms.aphiaRecordsByMatchNames([name])
    if res and res[0]:
        record = res[0][0]
        print("Record keys:", record.keys())
        # print("Full Record:", record)
        
        aphia_id = record['AphiaID']
        print(f"AphiaID: {aphia_id}")
        
        # Check direct distribution function if it exists in the library wrapper
        # The library is a wrapper around the REST API.
        # Let's check available functions in pyworms
        # print("pyworms attributes:", dir(pyworms))
        
        try:
            # The pyworms library might not have a direct helper for distributions,
            # but usually it wraps the common endpoints.
            # If not, we might need to query the REST API directly or check if getDistributions exists.
            if hasattr(pyworms, 'getAphiaDistributions'):
                dists = pyworms.getAphiaDistributions(aphia_id)
                print(f"Found {len(dists)} distribution records.")
                if dists:
                    print("Sample distribution:", dists[0])
            else:
                print("pyworms does not appear to have 'getAphiaDistributions'.")
                
        except Exception as e:
            print(f"Error checking distribution: {e}")

if __name__ == "__main__":
    check_species("Gadus morhua")
    check_species("Grimothea planipes")
