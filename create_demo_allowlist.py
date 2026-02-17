import pyworms
import json

families_to_check = [
    "Munididae",      # Grimothea
    "Eunicidae",      # Eunice
    "Potamotrygonidae", # Freshwater stingray
    "Cambaridae",     # Freshwater crayfish
    "Portunidae",     # Lissocarcinus
    "Oregoniidae",    # Hyas
    "Carcinidae",     # Carcinus
    "Cancridae"       # Dungeness crab
]

valid_marine = []
resp = pyworms.aphiaRecordsByMatchNames(families_to_check)

for name, records in zip(families_to_check, resp):
    if records:
        rec = records[0]
        im = rec.get('isMarine') in (1, True, '1')
        ib = rec.get('isBrackish') in (1, True, '1')
        print(f"{name}: Marine={im}, Brackish={ib}")
        if im or ib:
            valid_marine.append(name)
    else:
        print(f"{name}: Not found")

print(f"Valid Marine Families: {valid_marine}")

# Save to file
path = "assets/taxonomy/worms_marine_families_partial.json"
with open(path, 'w') as f:
    json.dump(valid_marine, f)
print(f"Saved partial list to {path}")
