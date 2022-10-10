# Code adapted from https://github.com/isl-org/MultiObjectiveOptimization/blob/master/multi_task/train_multi_task.py

import argparse
import json
from timeit import default_timer as timer
import torch
import random
import numpy as np
import wandb
import os
from supervised_experiments.utils import create_logger

import supervised_experiments.losses as losses_f
import supervised_experiments.datasets as datasets
import supervised_experiments.metrics as metrics
import supervised_experiments.model_selector as model_selector
from supervised_experiments.evaluate_multi_task import log_test_results

from optimizers.pcgrad import PCGrad
from optimizers.imtl import IMTL
from optimizers.mgda import MGDA
from optimizers.rlw import RLW
from optimizers.graddrop import GradDrop
from optimizers.baselines import Baseline


def save_model(models, optimizer, scheduler, tasks, epoch, args, folder="saved_models/", name="best"):

    if not os.path.exists(folder):
        os.makedirs(folder)

    state = {'epoch': epoch + 1,
             'model_rep': models['rep'].state_dict(),
             'optimizer_state': optimizer.state_dict(),
             'scheduler_state': scheduler.state_dict() if scheduler is not None else None,
             'args': vars(args)}
    for t in tasks:
        key_name = 'model_{}'.format(t)
        state[key_name] = models[t].state_dict()

    run_name = "debug" if args.debug else wandb.run.name
    torch.save(state, f"{folder}{args.label}_{run_name}_{name}_model.pkl")


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def test_evaluator(args, test_loader, tasks, DEVICE, model, loss_fn, metric, aggregators, logger, model_name, epoch):
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
    log_test_results(tasks, metric, losses, num_test_batches, aggregators, logger, do_wandb=not (args.debug),
                     model_name=model_name, epoch=epoch)


def train_multi_task(args, random_seed):

    # Set random seeds.
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)
    g = torch.Generator()
    g.manual_seed(random_seed)

    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    logger = create_logger('Main')
    if not args.debug:
        wandb.init(project="unitary_scalarization", group=args.label, config=args, reinit=True)
        dropout_str = "" if not args.dropout else "-dropout"
        wandb.run.name = f"{args.optimizer}{dropout_str}-lr:{args.lr}-wd:{args.weight_decay}_" + wandb.run.name

    with open(args.config_file) as config_params:
        configs = json.load(config_params)

    train_loader, val_loader = datasets.get_dataset(args.dataset, args.batch_size, configs,
                                                    generator=g, worker_init_fn=seed_worker, train=True)

    test_loader = datasets.get_dataset(args.dataset, args.batch_size, configs,
                                       generator=g, worker_init_fn=seed_worker, train=False)

    loss_fn = losses_f.get_loss(args.dataset, configs[args.dataset]['tasks'])
    metric, aggregators, model_saver = metrics.get_metrics(args.dataset, configs[args.dataset]['tasks'])

    model = model_selector.get_model(args.dataset, configs[args.dataset]['tasks'], device=DEVICE,
                                     add_dropout=args.dropout, no_dropout=args.no_dropout)
    model_params = [p for v in model.values() for p in list(v.parameters())]
    spec_params = [p for t in configs[args.dataset]['tasks'] for p in list(model[t].parameters())]

    optimizer = torch.optim.Adam(model_params, lr=args.lr, weight_decay=args.weight_decay)
    if args.decay_lr:
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.95)
    else:
        scheduler = None

    if args.optimizer == 'pcgrad':
        mtl_opt = PCGrad(optimizer)
    elif args.optimizer == "imtl":
        # Note that this operates on the gradient of the shared parameters by default (for RL, set ub=False)
        mtl_opt = IMTL(optimizer, spec_params)
    elif args.optimizer in ["mgda", "mgda-ub"]:
        mtl_opt = MGDA(optimizer, spec_params, normalize="loss+", ub=(args.optimizer == "mgda-ub"))
    elif args.optimizer in ["graddrop", "ran-graddrop"]:
        mtl_opt = GradDrop(optimizer, spec_params, use_sign=(args.optimizer == "graddrop"), p=args.p)
    elif "rlw" in args.optimizer:
        mtl_opt = RLW(optimizer, len(configs[args.dataset]['tasks']), distribution=args.optimizer.split("-")[1])
    else:
        if args.baseline_losses_weights:
            mtl_opt = Baseline(optimizer, weights=args.baseline_losses_weights)
        else:
            mtl_opt = Baseline(optimizer)

    best_result = {k: -float("inf") for k in model_saver}  # Something to maximize.
    train_val_stats = []  # saving validation stats per epoch
    tasks = configs[args.dataset]['tasks']
    n_iter = 0
    for epoch in range(args.num_epochs):
        start = timer()
        print('Epoch {} Started'.format(epoch))

        for m in model:
            model[m].train()

        losses_per_epoch = {t: 0.0 for t in tasks}
        for batch in train_loader:
            n_iter += 1
            # Read targets and images for the batch.
            images = batch[0].to(DEVICE)
            labels = {t: batch[i+1].to(DEVICE) for i, t in enumerate(tasks)}

            # Compute per-task losses.
            losses = []
            rep, _ = model['rep'](images, None)
            del images
            for t in tasks:
                out_t, _ = model[t](rep, None)
                # the losses are averaged within the MTL optimizers, possibly after manipulations per datapoint
                loss_t = loss_fn[t](out_t, labels[t], average=False)
                losses.append(loss_t)  # to backprop on
                losses_per_epoch[t] += loss_t.mean().detach()  # for logging purposes

            mtl_opt.iterate(losses, shared_repr=rep)

        if args.decay_lr:
            scheduler.step()

        if args.time_measurement_exp:
            # Measure time only, no need to log training/validation stats.
            epoch_runtime = timer() - start
            logger.info(f"epochs {epoch}/{args.num_epochs}: runtime: {epoch_runtime}")
            epoch_stats = {"runtime": epoch_runtime}
            if not args.debug:
                wandb.log(epoch_stats, step=epoch)
            train_val_stats.append(epoch_stats)
            continue

        # Print the stored (averaged across batches) training losses, per task.
        clog = "epochs {}/{}:".format(epoch, args.num_epochs)
        for t in tasks:
            clog += ' train_loss {} = {:5.4f}'.format(t, losses_per_epoch[t] / n_iter)
        logger.info(clog)
        epoch_stats = {}
        for i, t in enumerate(tasks):
            epoch_stats[f"train_loss_{t}"] = losses_per_epoch[t] / n_iter

        # Evaluate the model on the validation set.
        for m in model:
            model[m].eval()

        with torch.no_grad():
            losses_per_epoch = {t: 0.0 for t in tasks}
            num_val_batches = 0
            for batch_val in val_loader:
                val_images = batch_val[0].to(DEVICE)
                labels_val = {t: batch_val[i+1].to(DEVICE) for i, t in enumerate(tasks)}

                val_rep, _ = model['rep'](val_images, None)
                for t in tasks:
                    out_t_val, _ = model[t](val_rep, None)
                    loss_t = loss_fn[t](out_t_val, labels_val[t])
                    losses_per_epoch[t] += loss_t.item()  # for logging purposes
                    metric[t].update(out_t_val, labels_val[t])
                num_val_batches += 1

        # Print the stored (averaged across batches) validation losses and metrics, per task.
        clog = "epochs {}/{}:".format(epoch, args.num_epochs)
        metric_results = {}
        for t in tasks:
            metric_results[t] = metric[t].get_result()
            metric[t].reset()
            clog += ' val_loss {} = {:5.4f}'.format(t, losses_per_epoch[t] / num_val_batches)
            for metric_key in metric_results[t]:
                clog += ' val metric-{} {} = {:5.4f}'.format(metric_key, t, metric_results[t][metric_key])
            clog += " |||"

        # Store aggregator metrics (e.g., avg) as well
        for agg_key in aggregators:
            clog += ' val metric-{} = {:5.4f}'.format(agg_key, aggregators["avg"](metric_results))

        logger.info(clog)
        for i, t in enumerate(tasks):
            epoch_stats[f"val_loss_{t}"] = losses_per_epoch[t] / num_val_batches
            for metric_key in metric_results[t]:
                epoch_stats[f"val_metric_{metric_key}_{t}"] = metric_results[t][metric_key]

        # Store aggregator metrics (e.g., avg) as well
        for agg_key in aggregators:
            epoch_stats[f"val_metric_{agg_key}"] = aggregators[agg_key](metric_results)
        if not args.debug:
            wandb.log(epoch_stats, step=epoch)
        train_val_stats.append(epoch_stats)

        # Any time one of the model_saver metrics is improved upon, store a corresponding model.
        c_saver_metric = {k: model_saver[k](metric_results) for k in model_saver}
        for k in c_saver_metric:
            if c_saver_metric[k] >= best_result[k]:
                best_result[k] = c_saver_metric[k]
                # Evaluate the model on the test set and store relative results.
                test_evaluator(args, test_loader, tasks, DEVICE, model, loss_fn, metric, aggregators, logger, k, epoch)
                if args.store_models:
                    # Save (overwriting) any model that improves the average metric
                    save_model(model, optimizer, scheduler, tasks, epoch, args,
                               folder=configs["utils"]["model_storage"], name=k)

        end = timer()
        print('Epoch ended in {}s'.format(end - start))

    results_folder = configs["utils"]["results_storage"]
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)
    run_name = "debug" if args.debug else wandb.run.name
    torch.save({'stats': train_val_stats, 'args': vars(args)},
               f"{results_folder}{args.label}_{run_name}_validation_results.pkl")
    # Save training/validation results.
    if args.store_models and (not args.time_measurement_exp):
        # Save last model.
        save_model(model, optimizer, scheduler, tasks, epoch, args, folder=configs["utils"]["model_storage"],
                   name="last")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default='./dataset', help='Path to dataset folder')
    parser.add_argument('--label', type=str, default='', help='wandb group')
    parser.add_argument('--dataset', type=str, default='', help='which dataset to use',
                        choices=['celeba', 'mnist', 'cityscapes'])
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--p', type=float, default=0.1, help='Task dropout probability')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=100, help='Epochs to train for.')
    parser.add_argument('--optimizer', type=str, default='pcgrad', help='Optimiser to use',
                        choices=['pcgrad', "baseline", "imtl", "mgda", "mgda-ub", "graddrop", "ran-graddrop",
                                 "rlw-uniform", "rlw-normal", "rlw-dirichlet", "rlw-random_normal",
                                 "rlw-bernoulli", "rlw-constrained_bernoulli"])
    parser.add_argument('--debug', action='store_true', help='Debug mode: disables wandb.')
    parser.add_argument('--store_models', action='store_true', help='Whether to store  models at fixed frequency.')
    parser.add_argument('--decay_lr', action='store_true', help='Whether to decay the lr with the epochs.')
    parser.add_argument('--dropout', action='store_true', help='Whether to use additional dropout in training.')
    parser.add_argument('--no_dropout', action='store_true', help='Whether to not use dropout at all.')
    parser.add_argument('--weight_decay', type=float, default=0.0, help='L2 regularization.')
    parser.add_argument('--n_runs', type=int, default=1, help='Number of experiment repetitions.')
    parser.add_argument('--random_seed', type=int, default=1, help='Start random seed to employ for the run.')
    parser.add_argument('--config_file', type=str, default="supervised_experiments/configs.json")
    parser.add_argument('--baseline_losses_weights', type=int, nargs='+',
                        help='Weights to use for losses. Be sure that the ordering is correct! (ordering defined as in config for losses.')
    parser.add_argument('--time_measurement_exp', action='store_true',
                        help="whether to only measure time (does not log training/validation losses/metrics)")
    args = parser.parse_args()

    for i in range(args.n_runs):
        train_multi_task(args, args.random_seed + i)
