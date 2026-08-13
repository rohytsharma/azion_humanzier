#!/bin/sh
# Wait for pretraining to finish, then fine-tune on the rewrite pairs.
#
#   nohup ./run_finetune_after_pretrain.sh > finetune.log 2>&1 &
#
# The two jobs must not overlap: they would compete for the same GPU, and
# fine-tuning needs the finished weights as its starting point.

set -e
cd "$(dirname "$0")"

echo "waiting for pretraining to finish…"
while pgrep -f "training.pretrain" > /dev/null 2>&1; do
    sleep 120
done
echo "pretraining finished at $(date '+%Y-%m-%d %H:%M')"

if [ ! -f training/checkpoints/best.pt ]; then
    echo "no pretrained checkpoint — stopping" >&2
    exit 1
fi

if [ ! -f data/pairs/train.jsonl ]; then
    echo "building rewrite pairs…"
    .venv/bin/python -m data.make_pairs --pairs 150000 --val-pairs 3000
fi

echo "starting fine-tuning at $(date '+%Y-%m-%d %H:%M')"
.venv/bin/python -m training.finetune --epochs 2

echo "scoring the fine-tuned model at $(date '+%Y-%m-%d %H:%M')"
.venv/bin/python -m evaluation.evaluate_lm --split test --device cpu \
    --ckpt training/checkpoints/finetuned.pt || true

echo "done at $(date '+%Y-%m-%d %H:%M'). Run the app:"
echo "  .venv/bin/streamlit run app/streamlit_app.py"
