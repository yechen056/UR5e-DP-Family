from typing import Dict, List, Union

from model.common.module_attr_mixin import ModuleAttrMixin


class BaseSensoryEncoder(ModuleAttrMixin):
    def __init__(self):
        super().__init__()

    def forward(self, obs_dict: Union[Dict, List[Dict]]) -> Dict:
        raise NotImplementedError

    def modalities(self) -> List[str]:
        raise NotImplementedError

    def output_feature_dim(self) -> Dict:
        raise NotImplementedError
