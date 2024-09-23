class OMG(pl.LightningModule):

    def __init__(self, si, sampler, model):
        self.si = si 
        self.sampler = sampler
        self.model = model

    def forward(self, x, t)
        # preprocessing stuff
        x = self.model(x, t)
        return x

    def training_step(self, x_1)
        x_0 = self.sampler.sample_p_0() # this might need x_1 as input so number of atoms are consistent 
        t = self.sample_t() # maybe just directly call torch function 
        x_t = self.si.interpolate(x_1, x_0, t)
        model_prediction = self.model.predict(x_t, t)
        
        loss = self.si.loss(model_prediction, t, x_0, x_1)

        ground_truth = self.si.compute_ground_truth(x_1, x_0, t)
        pred = self.model(x_t, t)
        # record loss
        loss = mse.loss(pred, ground_truth) * self.si.derivative(...) # or self.si.loss(x, ground_truth)
       
        return loss

    # def validation_step(self, batch)

    def sample(self)
        # sample number of atoms beforehand 
        x_0 = self.sampler.sample()
        return self.si.integrate(x_0, self.model.de_fn)




