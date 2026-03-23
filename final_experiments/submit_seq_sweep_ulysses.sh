#!/bin/bash

# Sequence lengths to test
# SEQ_LENGTHS=(512 1024 2048 4096 8192 16384 32768)
SEQ_LENGTHS=(16384)

# Submit jobs for each sequence length
for SEQ_LEN in "${SEQ_LENGTHS[@]}"; do
    echo "Submitting job for sequence length: ${SEQ_LEN}"
    sbatch --export=ALL,SEQUENCE_LENGTH=${SEQ_LEN} train_multi_node_ulysses.sbatch
    sleep 5  # Small delay to avoid overwhelming the scheduler
done

echo "All jobs submitted!"