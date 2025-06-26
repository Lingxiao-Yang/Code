from amplicon_dataset import TemplateDataset
from pathlib import Path

class MeltDataset(TemplateDataset):
    def _convert(self, value: object):
        return str(value)