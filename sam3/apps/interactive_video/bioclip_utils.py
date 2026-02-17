from __future__ import annotations
import os
import shutil
import tempfile
import torch
import numpy as np
from PIL import Image
import json
from typing import List, Tuple, Optional, Any, Dict
from bioclip import TreeOfLifeClassifier, Rank

_BIOCLIP_CLASSIFIER = None

def load_bioclip_classifier(device: str = "cuda" if torch.cuda.is_available() else "cpu"):
    global _BIOCLIP_CLASSIFIER
    if _BIOCLIP_CLASSIFIER is None:
        print("Loading BioCLIP-2 TreeOfLifeClassifier...")
        _BIOCLIP_CLASSIFIER = TreeOfLifeClassifier()
        print("TreeOfLifeClassifier loaded. Applying marine/aquatic filter...")
        _apply_marine_filter(_BIOCLIP_CLASSIFIER)
        print("Marine filter applied.")
    return _BIOCLIP_CLASSIFIER

def _get_valid_names(classifier: TreeOfLifeClassifier, rank: Rank, names: List[str]) -> List[str]:
    """Returns only the names that exist in the classifier's label data for the given rank."""
    df = classifier.get_label_data()
    # Map Rank to column name
    col_map = {
        Rank.KINGDOM: 'kingdom',
        Rank.PHYLUM: 'phylum',
        Rank.CLASS: 'class',
        Rank.ORDER: 'order',
        Rank.FAMILY: 'family',
        Rank.GENUS: 'genus',
        Rank.SPECIES: 'species'
    }
    col = col_map.get(rank)
    if not col or col not in df.columns:
         print(f"Warning: Column '{col}' not found in BioCLIP label data.")
         return []
         
    valid_set = set(df[col].dropna().unique())
    # Check exact matches
    return [n for n in names if n in valid_set]

def _apply_marine_filter(classifier: TreeOfLifeClassifier):
    """
    Restricts the classifier to marine/aquatic taxa.
    
    1. Checks for 'assets/taxonomy/WoRMS_marine_families.json'.
       If found, applies STRICT filter based on this allowlist.
    2. If not found, uses heuristic strict filtering (Phyla/classes + exclusions).
    """
    
    # Try 1: Exact case
    allowlist_path = os.path.join(os.path.dirname(__file__), "../../../assets/taxonomy/WoRMS_marine_families.json")
    allowlist_path = os.path.abspath(allowlist_path)
    if not os.path.exists(allowlist_path):
        # Try 2: Lowercase
        allowlist_path = os.path.join(os.path.dirname(__file__), "../../../assets/taxonomy/WoRMS_marine_families.json")
        allowlist_path = os.path.abspath(allowlist_path)
    
    if os.path.exists(allowlist_path):
        print(f"Loading strict WoRMS family allowlist from {allowlist_path}...")
        with open(allowlist_path, 'r') as f:
            target_families = json.load(f)
            
        print(f"Allowlist contains {len(target_families)} families.")
        
        valid_families = _get_valid_names(classifier, Rank.FAMILY, target_families)
        print(f"BioCLIP intersection: {len(valid_families)} valid families.")
        
        if valid_families:
            mask = np.array(classifier.create_taxa_filter(Rank.FAMILY, valid_families))
            classifier.apply_filter(mask.tolist())
            print(f"Strict WoRMS filter applied. Retained {mask.sum()} taxa.")
            return
        else:
            print("Warning: No families from allowlist found in BioCLIP! Falling back to heuristics.")

    # Fallback Heuristics
    print("Applying heuristic marine filter...")
    
    # 1. Aquatic Phyla
    target_phyla = [
        # Animals
        "Porifera", "Ctenophora", "Cnidaria", "Echinodermata", 
        "Brachiopoda", "Bryozoa", "Nemertea", "Chaetognatha", 
        "Mollusca", "Annelida", "Platyhelminthes", "Hemichordata",
        "Xenacoelomorpha", "Rotifera", "Gastrotricha", 
        # Algae
        "Rhodophyta", "Chlorophyta", "Ochrophyta", "Haptophyta", 
        "Bacillariophyta", "Dinoflagellata"
    ]
    
    # 2. Aquatic Classes
    target_classes = [
        "Malacostraca", "Maxillopoda", "Cirripedia", "Thecostraca",
        "Copepoda", "Ostracoda", "Branchiopoda", "Pycnogonida",
        "Merostomata", "Xiphosura",
        "Ascidiacea", "Thaliacea", "Appendicularia",
        "Actinopterygii", "Chondrichthyes", "Myxini", "Petromyzontida",
        "Sarcopterygii", "Elasmobranchii", "Holocephali"
    ]
    
    # 3. Excluded Families
    excluded_families = [
        "Potamotrygonidae", "Cyprinidae", "Cichlidae", "Characidae", 
        "Loricariidae", "Poeciliidae", "Ranidae"
    ]

    valid_phyla = _get_valid_names(classifier, Rank.PHYLUM, target_phyla)
    valid_classes = _get_valid_names(classifier, Rank.CLASS, target_classes)
    valid_excluded = _get_valid_names(classifier, Rank.FAMILY, excluded_families)
    
    final_mask = None
    
    if valid_phyla:
        final_mask = np.array(classifier.create_taxa_filter(Rank.PHYLUM, valid_phyla))
        
    if valid_classes:
        mask_classes = np.array(classifier.create_taxa_filter(Rank.CLASS, valid_classes))
        if final_mask is None:
            final_mask = mask_classes
        else:
            final_mask = final_mask | mask_classes
            
    if valid_excluded:
        mask_excluded = np.array(classifier.create_taxa_filter(Rank.FAMILY, valid_excluded))
        if final_mask is not None:
             final_mask = final_mask & ~mask_excluded
             
    if final_mask is not None:
        classifier.apply_filter(final_mask.tolist())
        print(f"Heuristic filter applied. Retained {final_mask.sum()} taxa.")

def format_taxonomic_name(name: str) -> Tuple[str, str]:
    """
    Parses a scientific name (e.g. 'Carcinus maenas') into (Genus, Species) parts.
    Returns (Genus, Abbreviated) e.g. ('Carcinus', 'C. maenas').
    If not a binomial, returns (name, name).
    """
    parts = name.split()
    if len(parts) >= 2:
        genus = parts[0]
        species = parts[1]
        # Check if capitalized (scientific convention)
        if genus[0].isupper():
            abbrev = f"{genus[0]}. {species}"
            return genus, abbrev
    return name, name

import pyworms
import requests

# Simple memory cache for WoRMS
_WORMS_CACHE = {} # name -> bool (is_marine)
_WORMS_INFO_CACHE = {} # name -> dict (full record)
_WORMS_DIST_CACHE = {} # aphia_id -> list of distribution records

# GBIF Cache
_GBIF_CACHE_FILE = os.path.join(os.path.dirname(__file__), "../../../assets/taxonomy/gbif_cache.json")
_GBIF_CACHE = {} 

def _load_gbif_cache():
    global _GBIF_CACHE
    if os.path.exists(_GBIF_CACHE_FILE):
        try:
            with open(_GBIF_CACHE_FILE, 'r') as f:
                _GBIF_CACHE = json.load(f)
            # print(f"Loaded GBIF cache with {len(_GBIF_CACHE)} entries.")
        except Exception as e:
            print(f"Failed to load GBIF cache: {e}")

def _save_gbif_cache():
    try:
        os.makedirs(os.path.dirname(_GBIF_CACHE_FILE), exist_ok=True)
        with open(_GBIF_CACHE_FILE, 'w') as f:
            json.dump(_GBIF_CACHE, f, indent=2)
    except Exception as e:
        print(f"Failed to save GBIF cache: {e}")

_load_gbif_cache()


def validate_marine_species(name: str) -> bool:
    """
    Checks if a species name is a valid marine species using WoRMS.
    Returns True if marine/brackish, False otherwise.
    """
    # Check cache
    if name in _WORMS_CACHE:
        return _WORMS_CACHE[name]

    try:
        records = pyworms.aphiaRecordsByMatchNames([name])
        
        if records and records[0]:
            # Take the first match
            rec = records[0][0] # first name's first match
            
            # Cache full record for later use (e.g. regions)
            _WORMS_INFO_CACHE[name] = rec
            
            # Inspect record
            is_marine = rec.get('isMarine') in [1, True, '1']
            is_brackish = rec.get('isBrackish') in [1, True, '1']
            
            if is_marine or is_brackish:
                _WORMS_CACHE[name] = True
                return True
            else:
                 _WORMS_CACHE[name] = False
                 return False
        else:
            # Not in WoRMS
            _WORMS_CACHE[name] = False
            return False
    except Exception as e:
        print(f"WoRMS API error for {name}: {e}")
        return False
def validate_region(name: str, allowed_regions: List[str] = None) -> bool:
    """
    Checks if a species is present in the specified regions using GBIF occurrence data.
    Maps input regions (e.g. 'Canada', 'Pacific') to GBIF Country Codes ('CA', 'US').
    Returns True if species has >0 occurrences in the target countries.
    """
    if not allowed_regions:
        return True
        
    # 1. Map regions to GBIF Country Codes
    region_map = {
        "canada": "CA", "ca": "CA", "british columbia": "CA", "bc": "CA",
        "united states": "US", "usa": "US", "us": "US", "washington": "US", "wa": "US", "oregon": "US", "california": "US",
        "mexico": "MX", "mx": "MX"
    }
    target_codes = set()
    for r in allowed_regions:
        r_lower = r.lower().strip()
        if r_lower in region_map:
            target_codes.add(region_map[r_lower])
            
    # Heuristic: If "Pacific" is requested, assume CA + US west coast context
    if not target_codes and any("pacific" in r.lower() for r in allowed_regions):
         target_codes.add("CA")
         target_codes.add("US")
         
    if not target_codes:
        # No mappable country codes found (e.g. just "Atlantic" or unknown).
        # Fallback: Return True to avoid blocking valid things we can't check.
        return True

    # 2. Check GBIF
    try:
        # Check cache
        if name in _GBIF_CACHE:
            cached_counts = _GBIF_CACHE[name]
            total_occurrences = sum(cached_counts.get(code, 0) for code in target_codes)
            return total_occurrences > 0
            
        print(f"DEBUG: Checking GBIF for '{name}' in {target_codes}...")
            
        # Match Species
        match_url = "https://api.gbif.org/v1/species/match"
        params = {'name': name, 'kingdom': 'Animalia'}
        r = requests.get(match_url, params=params, timeout=5)
        if r.status_code != 200:
            print(f"DEBUG: Match API failed {r.status_code}")
            return True # Fail open
            
        data = r.json()
        usage_key = data.get('usageKey')
        if not usage_key:
            print(f"DEBUG: No usageKey for '{name}'")
            # Species not found in GBIF backbone? 
            # If WoRMS found it, it exists. GBIF missing it is rare.
            # Fail open.
            return True
        
        print(f"DEBUG: UsageKey {usage_key}")
            
        # Check Occurrences for each code
        code_counts = {}
        found_any = False
        
        occ_url = "https://api.gbif.org/v1/occurrence/search"
        
        # Optimization: We can stop as soon as we find 1.
        # But caching useful for future.
        
        for code in target_codes:
            occ_params = {
                'taxonKey': usage_key,
                'country': code,
                'limit': 0
            }
            r_occ = requests.get(occ_url, params=occ_params, timeout=5)
            if r_occ.status_code == 200:
                count = r_occ.json().get('count', 0)
                code_counts[code] = count
                print(f"DEBUG: Count {code}: {count}")
                if count > 0:
                    found_any = True
            else:
                print(f"DEBUG: Occ API failed for {code}: {r_occ.status_code}")
                code_counts[code] = 0
        
        _GBIF_CACHE[name] = code_counts
        _save_gbif_cache()
        print(f"DEBUG: Result {found_any}")
        return found_any
        
    except Exception as e:
        print(f"GBIF checking error for {name}: {e}")
        return True # Fail open

def predict_hierarchical(
    crops: List[np.ndarray | Image.Image],
    threshold_species: float = 0.40,
    threshold_genus: float = 0.30,
    threshold_family: float = 0.20,
    top_k_check: int = 5,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    allowed_regions: List[str] = None
) -> List[Tuple[str, float]]:
    """
    Predicts taxonomy using pybioclip TreeOfLifeClassifier with hierarchical reporting
    AND WoRMS validation + Optional Region Filtering.
    """
    if not crops:
        return []

    classifier = load_bioclip_classifier(device)
    
    temp_dir = tempfile.mkdtemp(prefix="bioclip_inputs_")
    paths = []
    
    try:
        for idx, crop in enumerate(crops):
            if isinstance(crop, np.ndarray):
                if crop.ndim == 3 and crop.shape[2] == 3:
                     img = Image.fromarray(crop)
                elif crop.ndim == 3 and crop.shape[2] == 4:
                     img = Image.fromarray(crop[..., :3])
                else:
                     img = Image.fromarray(crop)
            else:
                img = crop
            
            p = os.path.join(temp_dir, f"crop_{idx}.jpg")
            img.save(p)
            paths.append(p)
            
        # Get top K results to check
        raw_results = classifier.predict(paths, Rank.SPECIES, k=top_k_check)
        
        final_results = []
        
        def process_single_pred(preds: List[Dict[str, Any]]) -> Tuple[str, float]:
             if not preds:
                 return ("Unknown", 0.0)
             
             # Iterate through preds to find first WoRMS match
             valid_pred = None
             is_marine_flag = False
             in_region_flag = False
             
             # Keep track of the top prediction's marine/region status if no valid_pred is found
             top_pred_is_marine = False
             top_pred_in_region = False

             for p in preds:
                  name = p.get("species", "")
                  if not name: continue
                  
                  current_is_marine = validate_marine_species(name)
                  current_in_region = False
                  if current_is_marine:
                      if allowed_regions:
                          current_in_region = validate_region(name, allowed_regions)
                      else:
                          current_in_region = True # No region filter, so it's "in region"
                  
                  if p == preds[0]: # Store status for the very top prediction
                      top_pred_is_marine = current_is_marine
                      top_pred_in_region = current_in_region

                  if current_is_marine and current_in_region:
                       valid_pred = p
                       is_marine_flag = True
                       in_region_flag = True
                       break
             
             if not valid_pred:
                  # No marine/region match in top K.
                  top = preds[0]
                  top_score = top.get("score", 0.0)
                  top_name = top.get("species", "Unknown")
                  
                  # Determine the label based on why it was filtered
                  filter_reason = []
                  if not top_pred_is_marine:
                      filter_reason.append("Non-marine")
                  if allowed_regions and not top_pred_in_region:
                      filter_reason.append("Region-filtered")
                  
                  label_suffix = ""
                  if filter_reason:
                      label_suffix = f" ({'/'.join(filter_reason)})"

                  return {
                      "label": f"{top_name}{label_suffix}",
                      "score": top_score,
                      "rank": "filtered",
                      "species_pred": top_name,
                      "genus_pred": top.get("genus", ""),
                      "family_pred": top.get("family", ""),
                      "common_name": top.get("common_name", ""),
                      "is_marine": top_pred_is_marine,
                      "in_region": top_pred_in_region
                  }

             # Valid prediction found
             top_score = valid_pred.get("score", 0.0)
             top_name = valid_pred.get("species", "Unknown")
             genus, abbrev = format_taxonomic_name(top_name)
             family = valid_pred.get("family", "")
             
             base_info = {
                 "species_pred": top_name,
                 "genus_pred": valid_pred.get("genus", genus), # Use genus from valid_pred if available, else from format_taxonomic_name
                 "family_pred": family,
                 "common_name": valid_pred.get("common_name", ""),
                 "is_marine": is_marine_flag,
                 "in_region": in_region_flag
             }
             
             if top_score >= threshold_species:
                  return {**base_info, "label": abbrev, "score": top_score, "rank": "species"}
             
             # Genus fallback logic (using MAX score per genus)
             genus_scores = {}
             for p in preds:
                 nm = p.get("species", "")
                 sc = p.get("score", 0.0)
                 g, _ = format_taxonomic_name(nm)
                 # Check if this prediction is valid marine/region before counting it?
                 # Ideally yes, but for now we trust the top-K flow. 
                 # If we are falling back, we might be aggregating invalid species?
                 # But 'valid_pred' ensured we have at least one valid one.
                 # Let's keep it simple: aggregate scores from the preds list (which are just candidates).
                 # Wait, if we aggregate 'Canis lupus' into 'Canis', we might validate 'Canis'.
                 # But we only return it if it matches the 'valid_pred' genus?
                 # No, we should probably check the genus of 'valid_pred'.
                 pass
                 
             # Re-thinking: We want to support the case where the species score is low,
             # but we are confident in the Genus.
             # We assume 'valid_pred' is the correct TAXON, just maybe not confident in species.
             # So we check the score of the Genus of 'valid_pred'.
             
             # Calculate max score for the genus of valid_pred
             valid_genus = valid_pred.get("genus", genus)
             best_genus_score = 0.0
             
             # Also calculate max score for the family of valid_pred
             valid_family = valid_pred.get("family", "")
             best_family_score = 0.0
             
             for p in preds:
                  # Use MAX aggregation
                  sc = p.get("score", 0.0)
                  p_genus = p.get("genus", "")
                  if not p_genus:
                       p_genus, _ = format_taxonomic_name(p.get("species",""))
                  
                  if p_genus == valid_genus:
                       best_genus_score = max(best_genus_score, sc)
                       
                  p_family = p.get("family", "")
                  if p_family == valid_family:
                       best_family_score = max(best_family_score, sc)

             if best_genus_score >= threshold_genus:
                  return {**base_info, "label": f"{valid_genus} sp.", "score": best_genus_score, "rank": "genus"}

             if best_family_score >= threshold_family and valid_family:
                  return {**base_info, "label": f"{valid_family} (Fam.)", "score": best_family_score, "rank": "family"}
                  
             return {**base_info, "label": f"{abbrev} (?)", "score": top_score, "rank": "uncertain"}

        if not paths:
             return []
             
        # Normalize raw_results structure
        # If we passed N images, we expect N items.
        # If raw_results is List[Dict], it means N items, each is Top-1.
        # If raw_results is List[List[Dict]], it means N items, each is Top-K.
        
        normalized = []
        if isinstance(raw_results, list):
            if raw_results and isinstance(raw_results[0], dict) and 'score' in raw_results[0]:
                # This case implies raw_results is a list of top-1 predictions, one dict per image.
                # We need to wrap each dict in a list to make it a list of top-K predictions (where K=1).
                for r in raw_results:
                    normalized.append([r])
            elif raw_results and isinstance(raw_results[0], list):
                # This is already a list of lists of dicts (N images, each with top-K predictions)
                normalized = raw_results
            elif len(raw_results) == 0:
                # No results, create empty lists for each path
                normalized = [[] for _ in paths]
        else:
            # Handle cases where raw_results might not be a list (e.g., single dict if only one image and k=1)
            # This is a fallback, ideally classifier.predict always returns a list.
            if isinstance(raw_results, dict):
                normalized = [[raw_results]] # Wrap single dict in a list, then in another list for single image
            else:
                normalized = [[] for _ in paths] # Default to empty if unexpected structure

        # Now iterate normalized
        for preds_list in normalized:
             final_results.append(process_single_pred(preds_list))
                 
        return final_results
        
    finally:
        shutil.rmtree(temp_dir)

# Keeping legacy for compatibility if needed (unused)
def classify_crops(*args, **kwargs):
    print("Warning: classify_crops is deprecated. Use predict_hierarchical.")
    return []
