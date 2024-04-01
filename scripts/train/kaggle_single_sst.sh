python classifier.py \
    --sst_train 'data/ids-sst-train.csv' \
    --sst_dev 'data/ids-sst-dev.csv' \
    --sst_test 'data/ids-sst-test.csv' \
    --batch_size 64 \
    --option pretrain \
    --lr 1e-3 \
    --use_gpu