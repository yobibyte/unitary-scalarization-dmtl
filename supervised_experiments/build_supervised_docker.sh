if [$1]; then
    # build for cpu, assuming we have gpu by default and go to the second branch
    cuda=cpu
else
    cuda=cu101 # dockerfile has it
fi
echo "${WANDB_API_KEY}"
docker build --build-arg uid=$UID --build-arg user=$USER --build-arg cuda=$cuda --build-arg wandb_api_key="${WANDB_API_KEY}" -t mtrlsupervised ./supervised_experiments/
