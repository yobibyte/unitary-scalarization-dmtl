if [ $1 = "cpu" ]; then
    docker run -it -v `pwd`:/$USER/mtrlsupervised:rw --hostname $HOSTNAME --workdir /$USER/mtrlsupervised mtrlsupervised 
else
    docker run --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=$1 -it -v `pwd`:/$USER/mtrlsupervised:rw -v $DATA_FOLDER:/data:rw  --hostname $HOSTNAME --workdir /$USER/mtrlsupervised mtrlsupervised
fi
