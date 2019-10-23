import json
from mff import models

def load(filename):
    """ Load GP module based on train mode, kernel and number of atomic species.
    """
    
    with open(filename) as json_file:  
        metadata = json.load(json_file)
        
    model = metadata['model']
    if model == "TwoBodySingleSpeciesModel":
        m = models.TwoBodySingleSpeciesModel.from_json(filename)
    elif model == "ThreeBodySingleSpeciesModel":
        m = models.ThreeBodySingleSpeciesModel.from_json(filename)
    elif model == "CombinedSingleSpeciesModel":
        m = models.CombinedSingleSpeciesModel.from_json(filename)
    elif model == "TwoBodyTwoSpeciesModel":
        m = models.TwoBodyTwoSpeciesModel.from_json(filename)
    elif model == "ThreeBodyTwoSpeciesModel":
        m = models.ThreeBodyTwoSpeciesModel.from_json(filename)
    elif model == "CombinedTwoSpeciesModel":
        m = models.CombinedTwoSpeciesModel.from_json(filename)
    elif model == "TwoBodyManySpeciesModel":
        m = models.TwoBodyManySpeciesModel.from_json(filename)
    elif model == "ThreeBodyManySpeciesModel":
        m = models.ThreeBodyManySpeciesModel.from_json(filename)
    elif model == "CombinedManySpeciesModel":
        m = models.CombinedManySpeciesModel.from_json(filename)
    else:
        print("Json file does contain unexpected model name")
        return 0
    return m