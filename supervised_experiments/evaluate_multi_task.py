import argparse
import json
import torch
import random
import numpy as np
import os
import wandb
from supervised_experiments.utils import create_logger

import supervised_experiments.losses as losses_f
import supervised_experiments.datasets as datasets
import supervised_experiments.metrics as metrics
import supervised_experiments.model_selector as model_selector


def load_saved_model(models, tasks, net_basename, folder="saved_models/", name="best"):
    state = torch.load(f"{folder}{net_basename}_{name}_model.pkl")
    models['rep'].load_state_dict(state["model_rep"])
    for t in tasks:
        models[t].load_state_dict(state[f"model_{t}"])


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def log_test_results(tasks, metric, losses, num_test_batches, aggregators, logger, do_wandb=False,
                     model_name=None, epoch=-1):
    spaced_modelname = model_name + " " if model_name is not None else ""
    underscored_modelname = model_name + "_" if model_name is not None else ""
    clog = ""
    metric_results = {}
    for t in tasks:
        metric_results[t] = metric[t].get_result()
        metric[t].reset()
        clog += ' {}test_loss {} = {:5.4f}'.format(spaced_modelname, t, losses[t] / num_test_batches)
        for metric_key in metric_results[t]:
            clog += ' {}test metric-{} {} = {:5.4f}'.format(spaced_modelname, metric_key, t, metric_results[t][metric_key])
        clog += " |||"

    for agg_key in aggregators.keys():
        clog += ' {}test metric-{} = {:5.4f}'.format(spaced_modelname, agg_key, aggregators[agg_key](metric_results))

    logger.info(clog)
    test_stats = {}
    for i, t in enumerate(tasks):
        test_stats[f"{underscored_modelname}test_loss_{t}"] = losses[t] / num_test_batches
        for metric_key in metric_results[t]:
            test_stats[f"{underscored_modelname}test_metric_{metric_key}_{t}"] = metric_results[t][metric_key]

    for agg_key in aggregators.keys():
        test_stats[f"{underscored_modelname}test_metric_{agg_key}"] = aggregators[agg_key](metric_results)

    if do_wandb:
        wandb.log(test_stats, step=epoch)

    return test_stats


def test_multi_task(args, random_seed):

    # Set random seeds.
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)
    g = torch.Generator()
    g.manual_seed(random_seed)

    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    logger = create_logger('Main')
    with open('supervised_experiments/configs.json') as config_params:
        configs = json.load(config_params)

    test_loader = datasets.get_dataset(args.dataset, args.batch_size, configs,
                                       generator=g, worker_init_fn=seed_worker, train=False)
    loss_fn = losses_f.get_loss(args.dataset, configs[args.dataset]['tasks'])
    metric, aggregators, _ = metrics.get_metrics(args.dataset, configs[args.dataset]['tasks'])

    model = model_selector.get_model(args.dataset, configs[args.dataset]['tasks'], device=DEVICE)
    tasks = configs[args.dataset]['tasks']
    load_saved_model(model, tasks, args.net_basename, folder=configs["utils"]["model_storage"], name=args.model_type)

    # Evaluate the model on the test set.
    for m in model:
        model[m].eval()

    with torch.no_grad():
        losses = {t: 0.0 for t in tasks}
        num_test_batches = 0
        for batch_val in test_loader:
            test_images = batch_val[0].to(DEVICE)
            test_labels = {t: batch_val[i+1].to(DEVICE) for i, t in enumerate(tasks)}

            val_rep, _ = model['rep'](test_images, None)
            for t in tasks:
                out_t_val, _ = model[t](val_rep, None)
                loss_t = loss_fn[t](out_t_val, test_labels[t])
                losses[t] += loss_t.item()  # for logging purposes
                metric[t].update(out_t_val, test_labels[t])
            num_test_batches += 1

    print(f"Model {args.model_type}")
    # Print the stored (averaged across batches) test losses and metrics, per task.
    test_stats = log_test_results(tasks, metric, losses, num_test_batches, aggregators, logger)

    # Save test results.
    results_folder = configs["utils"]["results_storage"]
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)
    torch.save({'stats': test_stats, 'args': vars(args)},
               f"{results_folder}{args.net_basename}_{args.model_type}_test_results.pkl")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--net_basename', type=str, default='', help='basename of network (excludes _x_model.pkl)')
    parser.add_argument('--dataset', type=str, default='', help='which dataset to use', choices=['celeba', 'mnist'])
    parser.add_argument('--model_type', type=str, default='best', help='best or last model', choices=['best', 'last'])
    parser.add_argument('--random_seed', type=int, default=1, help='Start random seed to employ for the run.')
    parser.add_argument('--config_file', type=str, default="supervised_experiments/configs.json")
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size')

    args = parser.parse_args()

    test_multi_task(args, args.random_seed)
