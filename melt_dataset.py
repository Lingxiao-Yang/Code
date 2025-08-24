from pathlib import Path
from typing import Dict, List, Optional
import re
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
from scipy import signal

from process_all import list_subfolders
from melt_parser import find_xlsx_in,Melt, plot_melt_wave

ROOT=Path("Amplicon Length/Melt Dataset/")

STANDARD_TEMP= np.arange(58, 98.2, 0.2).tolist()

Melt_DataSet={}

class MeltDataset:
    _REGION_SHEETS = ("16s-23s", "23s-5s", "Thr-Tyr")
    def __init__(self, folder: str | Path):
        xlsx_path = find_xlsx_in(folder)[0]
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

        #self._data: Dict[str, Dict[str, Dict[str,List[float]]]] = {n: {region:{} for region in self._REGION_SHEETS} for n in names}
        #self._data:Dict[str,QPCR]= {name:QPCR("Dataset", name) for name in names}
        self._data: Dict[str, Melt]={name:Melt(f"Simulated {name}", name) for name in names}
        subfolders=list_subfolders(folder)

        for subfolder in subfolders:
            xml_files = sorted(subfolder.glob("*.txt"))
            for xml_file in xml_files:
                try:
                    x, y, i, j = extract_xml(xml_file)
                    region= [r for r in self._REGION_SHEETS if r in subfolder.name][0]
                    self._data[names[i-1]].add_wave("RFU", region, y)
                    self._data[names[i-1]].add_wave("temp",region,x)

                except Exception as e:
                    print(f"Error processing {xml_file}: {e}")

        self.bacteria_names: List[str] = names

    def get_profile(
            self, name: str, region: Optional[str] = None
    ) -> Melt:
        key = name.strip()
        try:
            entry = self._data[key]
        except KeyError as exc:
            raise KeyError(f"Bacterium '{name}' not found") from exc

        if region is None:
            return entry
        if region not in self._REGION_SHEETS:
            raise ValueError(f"Unknown region: {region!r}")
        return entry

    def __getitem__(self, name: str):
        return self.get_profile(name)

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"{len(self.bacteria_names)} bacteria * {len(self._REGION_SHEETS)} regions)"
        )

def extract_indices(filename: str):
    match=re.match(r"ROW\{(\d+)}COL\{(\d+)}_Melt", filename)
    if not match:
        raise ValueError(f"Filename '{filename}' does not match expected pattern.")
    i,j=map(int, match.groups())
    return i, j

def text_to_list(text: str)-> list[float]:
    if text:
        return [float(x) for x in text.split(" ") if x.strip()]
    return []

def extract_xml(filename: Path, need_indices: bool = True):
    with open(filename, "r", encoding="utf-8") as file:
        tree = ET.parse(file)
    root = tree.getroot()
    temp= root.find(".//temperature")
    helicity= root.find(".//helicity")

    temp=np.array(text_to_list(temp.text) if temp is not None else [])
    helicity=np.array(text_to_list(helicity.text) if helicity is not None else [])
    if temp.size==0 or helicity.size==0:
        raise ValueError(f"Missing temperature or helicity data in {filename}")

    helicity=np.interp(STANDARD_TEMP, temp, helicity)
    temp=STANDARD_TEMP

    helicity=signal.savgol_filter(helicity, 5, 2)

    if need_indices:
        i, j = extract_indices(filename.name)
    else:
        i, j = 0, 0
    return temp,helicity, i,j