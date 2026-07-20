import torch.optim as optim


def create_optimizer(model, learning_rate, weight_decay):

    return optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=weight_decay,
    )