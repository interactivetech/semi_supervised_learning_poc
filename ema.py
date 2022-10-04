class EMA:
    """
    EMA model
    Implementation from https://fyubang.com/2019/06/01/ema/
    """

    def __init__(self, model, decay):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}

    def load(self, ema_model):
        for name, param in ema_model.named_parameters():
            self.shadow[name] = param.data.clone()

    def register(self):
        for name, param in self.model.named_parameters():
            self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
            self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            self.backup[name] = param.data
            param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            param.data = self.backup[name]
        self.backup = {}

def set_ema_model(ema_model,model):
    """
    initialize ema model from model
    """
    # ema_model = self.net_builder(num_classes=self.num_classes)
    ema_model.load_state_dict(model.state_dict())
    return ema_model

class EMADriver:
    def __init__(self,model,ema_model,ema_m = 0.9):
        super().__init__()
        self.model = model
        self.ema_model = ema_model
        self.ema_m = ema_m
    def before_run(self):
        self.ema = EMA(self.model, self.ema_m)
        self.ema.register()
        # if algorithm.resume == True:
        #     algorithm.ema.load(algorithm.ema_model)

    def after_train_step(self):
        if self.ema is not None:
            self.ema.update()
            self.ema_model.load_state_dict(self.model.state_dict())
            self.ema_model.load_state_dict(self.ema.shadow, strict=False)