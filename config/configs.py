import os

class dataConfig:
    def __init__(self,config_dict) -> None:
        print(config_dict)
        for k,v in config_dict.items():
            self.__setattr__(k,v)
