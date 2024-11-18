from .head import Head


class PassThrough(Head):

    def __init__(self):
        super().__init__() 

    def forward(self, x, t, prop=None):
        return x

    def enable_masked_species(self) -> None:
        """
        Enable a masked species (with token 0) in the head.
        """
        pass
