python3.8 supervised_experiments/train_multi_task.py --label mnist_testeval_baseline --optimizer baseline --dataset mnist --lr 0.01 --decay_lr --num_epochs 100 --store_models --n_runs 10
python3.8 supervised_experiments/train_multi_task.py --label mnist_testeval_imtl --optimizer imtl --dataset mnist --lr 0.01 --decay_lr --num_epochs 100 --store_models --n_runs 10
python3.8 supervised_experiments/train_multi_task.py --label mnist_testeval_mgda-ub --optimizer mgda-ub --dataset mnist --lr 0.01 --num_epochs 100 --store_models --decay_lr --n_runs 10
python3.8 supervised_experiments/train_multi_task.py --label mnist_testeval_graddrop --optimizer graddrop --dataset mnist --lr 0.01 --decay_lr --num_epochs 100 --store_models --n_runs 10
python3.8 supervised_experiments/train_multi_task.py --label mnist_testeval_pcgrad --optimizer pcgrad --dataset mnist --lr 0.01 --decay_lr --num_epochs 100 --store_models --n_runs 10
python3.8 supervised_experiments/train_multi_task.py --label mnist_testeval_rlw-dirichlet --optimizer rlw-dirichlet --dataset mnist --lr 0.01 --decay_lr --num_epochs 100 --store_models --n_runs 10
python3.8 supervised_experiments/train_multi_task.py --label mnist_testeval_rlw-normal --optimizer rlw-normal --dataset mnist --lr 0.01 --decay_lr --num_epochs 100 --store_models --n_runs 10

python3.8 supervised_experiments/train_multi_task.py --label mnist_time_baseline --optimizer baseline --dataset mnist --lr 0.01 --decay_lr --num_epochs 1 --store_models --n_runs 10 --time_measurement_exp
python3.8 supervised_experiments/train_multi_task.py --label mnist_time_imtl --optimizer imtl --dataset mnist --lr 0.01 --decay_lr --num_epochs 1 --store_models --n_runs 10 --time_measurement_exp
python3.8 supervised_experiments/train_multi_task.py --label mnist_time_mgda-ub --optimizer mgda-ub --dataset mnist --lr 0.01 --num_epochs 1 --store_models --decay_lr --n_runs 10 --time_measurement_exp
python3.8 supervised_experiments/train_multi_task.py --label mnist_time_graddrop --optimizer graddrop --dataset mnist --lr 0.01 --decay_lr --num_epochs 1 --store_models --n_runs 10 --time_measurement_exp
python3.8 supervised_experiments/train_multi_task.py --label mnist_time_pcgrad --optimizer pcgrad --dataset mnist --lr 0.01 --decay_lr --num_epochs 1 --store_models --n_runs 10 --time_measurement_exp
python3.8 supervised_experiments/train_multi_task.py --label mnist_time_rlw-dirichlet --optimizer rlw-dirichlet --dataset mnist --lr 0.01 --decay_lr --num_epochs 1 --store_models --n_runs 10 --time_measurement_exp
python3.8 supervised_experiments/train_multi_task.py --label mnist_time_rlw-normal --optimizer rlw-normal --dataset mnist --lr 0.01 --decay_lr --num_epochs 1 --store_models --n_runs 10 --time_measurement_exp