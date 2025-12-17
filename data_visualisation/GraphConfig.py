from dataclasses import dataclass
@dataclass
class GraphConfig:
    def __init__(self,  target:str, title:str, xlabel:str, _1x1_restriction:int, _2x2_restriction:int, _3x3_restriction:int):
        self.target = target
        self.title = title
        self.xlabel = xlabel
        self.x1x1 = _1x1_restriction
        self.x2x2 = _2x2_restriction
        self.x3x3 = _3x3_restriction