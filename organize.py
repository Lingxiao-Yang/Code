import pandas as pd
from pathlib import Path

# Paths
reordered_path = Path("./Thermosleuth gen3 simulation/16s23s-189 bacteria thermosleuth_reordered.xlsx")
profile_path   = Path("./Amplicon Length/Final Amplicon Profile.xlsx")

# Sheet names
reordered_sheet = "Reordered Amplicons"
order_sheet     = "Bacteria"
profile_sheet   = "16s-23s"

# 1) Load your reordered simulated amplicons
df_sim = pd.read_excel(reordered_path, sheet_name=reordered_sheet)
species_col = df_sim.columns[0]
df_sim = df_sim.set_index(species_col)

# 2) Load the ordered species list (so we know which profile‐row goes with which name)
df_order = pd.read_excel(profile_path, sheet_name=order_sheet, header=None)
order_list = df_order.iloc[:,0].tolist()

# 3) Load the profile lengths (no header, no names)
df_prof = pd.read_excel(profile_path, sheet_name=profile_sheet, header=None)

# Drop any all‐NaN columns, then assign the species index
df_prof = df_prof.dropna(axis=1, how="all")
df_prof.index = order_list

# 4) Compare row‐by‐row
results = []
for species in df_sim.index:
    sim_lengths  = df_sim.loc[species].dropna().astype(int).tolist()
    prof_lengths = df_prof.loc[species].dropna().astype(int).tolist()
    results.append({
        "Species":      species,
        "Simulated":    sim_lengths,
        "Profile":      prof_lengths,
        "Exact match":  sim_lengths == prof_lengths
    })

df_compare = pd.DataFrame(results)

# 5) Report mismatches
mismatches = df_compare[~df_compare["Exact match"]]
if not mismatches.empty:
    print("MISMATCHES FOUND:")
    print(mismatches[["Species","Simulated","Profile"]].to_string(index=False))
else:
    print("All rows match exactly!")

# 6) (Optional) Save full comparison
out_path = reordered_path.with_name("amplicon_length_comparison.xlsx")
with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
    df_compare.to_excel(writer, sheet_name="Comparison", index=False)

print(f"\nComparison table written to {out_path}")
