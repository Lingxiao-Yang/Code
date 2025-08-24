from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd

class AmpliconDataset:
    """
    Load and serve bacterial amplicon-profile data stored in an Excel workbook.

    Workbook layout
    ---------------
    • Sheet “Bacteria”   : names of the 189 bacteria in cells A1-A189
    • Sheet “16s-23s”    : amplicon lengths for the 16S-23S rDNA region
    • Sheet “23s-5s”     : amplicon lengths for the 23S-5S rDNA region
    • Sheet “Thr-Tyr”    : amplicon lengths for the Thr-Tyr intergenic region

    Example
    -------
    >>> ds = AmpliconDataset("amplicon_profiles.xlsx")
    >>> ds["Yersinia pestis"]
    {'16s-23s': [728, 980, 1050],
     '23s-5s' : [],
     'Thr-Tyr': [482]}
    >>> ds.get_profile("Salmonella enterica", region="23s-5s")
    [396, 402, 402]
    """

    _REGION_SHEETS = ("16s-23s", "23s-5s", "Thr-Tyr")

    def __init__(self, xlsx_path: str | Path):
        xlsx_path = Path(xlsx_path).expanduser().resolve()
        if not xlsx_path.exists():
            raise FileNotFoundError(xlsx_path)

        sheets: dict[str, pd.DataFrame] = pd.read_excel(
            xlsx_path, sheet_name=["Bacteria", *self._REGION_SHEETS], header=None
        )

        names: List[str] = (
            sheets["Bacteria"].iloc[:, 0].astype(str).str.strip().tolist()
        )
        if len(names) != 189:
            raise ValueError(
                f"Expected 189 bacteria in sheet 'Bacteria', got {len(names)}"
            )

        self._data: Dict[str, Dict[str, List[int]]] = {n: {} for n in names}

        for region in self._REGION_SHEETS:
            df = sheets[region]

            if df.shape[0] < 189:
                df = df.reindex(range(189))

            for idx, (name, row) in enumerate(zip(names, df.itertuples(index=False))):
                values = [int(x) for x in row if pd.notna(x)]
                self._data[name][region] = values

        self.bacteria_names: List[str] = names

    def get_profile(
        self, name: str, region: Optional[str] = None
    ) -> Dict[str, List[int]] | List[int]:
        key = name.strip()
        try:
            entry = self._data[key]
        except KeyError as exc:
            raise KeyError(f"Bacterium '{name}' not found") from exc

        if region is None:
            return entry
        if region not in self._REGION_SHEETS:
            raise ValueError(f"Unknown region: {region!r}")
        return entry[region]


    def __getitem__(self, name: str):
        return self.get_profile(name)

    def __len__(self):
        """Return the number of bacteria in the dataset."""
        return len(self.bacteria_names)

    def size(self):
        """Return the size of the dataset (number of bacteria)."""
        return len(self.bacteria_names)

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"{len(self.bacteria_names)} bacteria * {len(self._REGION_SHEETS)} regions)"
        )
