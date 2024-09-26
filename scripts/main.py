from lightning.pytorch.cli import LightningCLI
#from omg.omg import OMG
#from ?data stuff here?
from omg.model.model import Model 

def main():
    #cli = LightningCLI(OMG, ?Data?)
    cli = LightningCLI(Model, run=False)

if __name__ == "__main__":
    main()

