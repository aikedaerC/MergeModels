import torch
import os 
import numpy as np

def save_checkpoint(model, args, path):
    # Create the directory if it doesn't exist
    os.makedirs(os.path.dirname(path), exist_ok=True)
    checkpoint = {'model': model.state_dict(), 'args': args}
    torch.save(checkpoint, path)

def load_checkpoint(create_model, load_data, logger, path):
    checkpoint = torch.load(path)
    args = checkpoint['args']

    logger.info(args)
    # Avoid randomness of cuda, but it will slow down the training
    if "cifar" in args.dataset:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    device = torch.device("cuda:" + str(args.gpu) if torch.cuda.is_available() else "cpu")
    logger.info(device)

    torch.set_printoptions(threshold=np.inf)

    # load dataset
    args.balance_fintune = True
    dataset = load_data(args, args.dataset)
    # create model
    model = create_model(args, model_name=args.model, output_dim=dataset[7])
    model.load_state_dict(checkpoint['model'])
    return dataset, model, args, device