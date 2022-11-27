import utils.argparser as argparser
import utils.embedding as embedding
import data.dataset as dataset_factory
import models.model as model_factory
import train.train_model as train
import pickle
import os

if __name__ == '__main__':
    # update and print args
    args = argparser.parse_args()

    embeddings, token2id = embedding.get_embedding_tensor(args)

    train_data, valid_data, test_data = dataset_factory.get_dataset(args, token2id)

    rationale_data = dataset_factory.get_rationale(args, token2id)

    gen, model = model_factory.get_model(args, embeddings)

    save_path = args.results_path

    args.model_path = os.path.join(save_path, f"{args.dataset}")

    if args.train :
        epoch_stats, model, gen = train.train_model(train_data, valid_data, model, gen, args)
        args.epoch_stats = epoch_stats
        print("Save train/dev results to", save_path)
        args_dict = vars(args)
        pickle.dump(args_dict, open(os.path.join(save_path, f'{args.dataset}_train_args_dict.pkl'),'wb') )

    if args.test :
        test_stats = train.test_model(test_data, model, gen, args)
        args.test_stats = test_stats
        args.train_data = train_data
        args.test_data = test_data
        print("Save test results to", save_path)
        args_dict = vars(args)
        pickle.dump(args_dict, open(os.path.join(save_path, f'{args.dataset}_test_args_dict.pkl'),'wb') )

    if args.get_rationales:
        rationale_stats = train.test_model(rationale_data, model, gen, args)
        args.test_stats = rationale_stats
        args.train_data = train_data
        args.test_data = rationale_data

        print("Save rationale results to", save_path)
        args_dict = vars(args)
        pickle.dump(args_dict, open(os.path.join(save_path, f'{args.dataset}_rationale_args_dict.pkl'),'wb') )
