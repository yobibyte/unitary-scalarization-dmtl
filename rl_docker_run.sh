if [ $1 = "cpu" ]; then
    docker run -it -v `pwd`:/$USER/rlmtl:rw --hostname $HOSTNAME --workdir /$USER/rlmtl/rl_experiments/mtrl rlmtl
else
    docker run --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=$1 -it -v `pwd`:/$USER/rlmtl:rw --hostname $HOSTNAME --workdir /$USER/rlmtl/rl_experiments/mtrl rlmtl
fi
