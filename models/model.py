import torch
import models.encoder as encoder
import models.generator as generator
import utils.learn as learn
import os

def get_model(args, embeddings):
    if args.snapshot is None:
        gen   = generator.Generator(embeddings, args)
        model = encoder.Encoder(embeddings, args)
    else :
        print('\nLoading model from [%s]...' % args.snapshot)
        try:
            gen_path = learn.get_gen_path(args.snapshot)
            if os.path.exists(gen_path):
                gen = torch.load(gen_path)
            model = torch.load(args.snapshot)
        except :
            print("Sorry, This snapshot doesn't exist."); exit()

    if args.num_gpus > 1:
        model = torch.nn.DataParallel(model,
                                    device_ids=range(args.num_gpus))

        if not gen is None:
            gen = torch.nn.DataParallel(gen,
                                    device_ids=range(args.num_gpus))
    return gen, model