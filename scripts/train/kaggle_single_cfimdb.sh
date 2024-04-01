python classifier.py \
    --cfimdb_train 'data/ids-cfimdb-train.csv' \
    --cfimdb_dev 'data/ids-cfimdb-dev.csv' \
    --cfimdb_test 'data/ids-cfimdb-test.csv' \
    --batch_size 64 \
    --option pretrain \
    --lr 1e-3 \
    --use_gpu