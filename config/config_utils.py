#Define the class ExperimentConfig. This class parses a .yaml file and sets its attributes as the parameters defined in the file.
import yaml

class ExperimentConfig:
   def __init__(self, input_config):
       if isinstance(input_config, str):
           self.config = ExperimentConfig.parse_yaml(input_config)
       else:
           self.config = input_config
            
       self._parse_config()
   
   @staticmethod
   def parse_yaml(file):
        with open(file, 'r') as file:
            return yaml.safe_load(file)
   
   def _parse_config(self):
       
       for key, value in self.config.items():
           #If the value is a dict, we need to create an aditional class in the key
              if isinstance(value, dict):
                setattr(self, key, ExperimentConfig(value))
              else:
                setattr(self, key, value)

