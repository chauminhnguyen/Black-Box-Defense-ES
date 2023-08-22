from torch.optim import Adam, SGD

def build_opt(optimizer_method, models):
    f = lambda models: models.parameters()
    if optimizer_method.lower() =='adam':
        optimizer = Adam(denoiser.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif optimizer_method.lower() =='sgd':
        optimizer = SGD(itertools.chain(denoiser.parameters(), encoder.parameters()), lr=args.lr, weight_decay=args.weight_decay)
    return optimizer

