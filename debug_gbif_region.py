
import requests

def check_gbif_us(species_name):
    print(f"Checking {species_name} in US...")
    
    # 1. Match
    match_url = "https://api.gbif.org/v1/species/match"
    params = {'name': species_name, 'kingdom': 'Animalia'}
    r = requests.get(match_url, params=params)
    data = r.json()
    usage_key = data.get('usageKey')
    if not usage_key:
        print("Species not found.")
        return

    print(f"Key: {usage_key}")

    # 2. Check US
    occ_url = "https://api.gbif.org/v1/occurrence/search"
    occ_params = {'taxonKey': usage_key, 'country': 'US', 'limit': 5}
    r_occ = requests.get(occ_url, params=occ_params)
    data_occ = r_occ.json()
    count = data_occ.get('count', 0)
    print(f"US Count: {count}")
    
    for res in data_occ.get('results', []):
        print(f"  - State: {res.get('stateProvince')} | Loc: {res.get('decimalLatitude')}, {res.get('decimalLongitude')}")

if __name__ == "__main__":
    check_gbif_us("Doclea brachyrhynchos")
