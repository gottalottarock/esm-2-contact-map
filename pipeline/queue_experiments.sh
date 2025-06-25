dvc exp run $@ --queue -S "datamodule=base_allseq" -n "unsup_allseq"
dvc exp run $@ --queue -S "datamodule=base_allseq_exc_cluster" -n "unsup_allseq_exc_cluster"
dvc queue status