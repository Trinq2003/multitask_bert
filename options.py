from argparse import ArgumentParser

def classifier_get_args():
    parser = ArgumentParser(description='Arguments for the single-task classifier training')
    parser.add_argument("--seed", type=int, default=11711)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--option", type=str,
                        help='pretrain: the BERT parameters are frozen; finetune: BERT parameters are updated',
                        choices=('pretrain', 'finetune'), default="pretrain")
    parser.add_argument("--use_gpu", action='store_true')
    parser.add_argument("--dev_out", type=str, default="cfimdb-dev-output.txt")
    parser.add_argument("--test_out", type=str, default="cfimdb-test-output.txt")

    parser.add_argument("--batch_size", help='sst: 64, cfimdb: 8 can fit a 12GB GPU', type=int, default=8)
    parser.add_argument("--hidden_dropout_prob", type=float, default=0.3)
    parser.add_argument("--lr", type=float, help="learning rate, default lr for 'pretrain': 1e-3, 'finetune': 1e-5", default=1e-5)
    parser.add_argument("--extension", type=str, default="default")

    # datapath
    parser.add_argument('--sst_train')
    parser.add_argument('--sst_dev')
    parser.add_argument('--sst_test')

    parser.add_argument('--cfimdb_train')
    parser.add_argument('--cfimdb_dev')
    parser.add_argument('--cfimdb_test')

    # adversarial regularization
    parser.add_argument('--pgd_k', type=int, default=1)
    parser.add_argument('--pgd_epsilon', type=float, default=1e-5)
    parser.add_argument('--pgd_lambda', type=float, default=1)

    # bergman momentum
    parser.add_argument('--mbpp_beta', type=float, default=0.995)
    parser.add_argument('--mbpp_mu', type=float, default=1)


    args = parser.parse_args()
    return args

def multitask_classifier_get_args():
    parser = ArgumentParser(description='Arguments for the multi-task classifier training')
    # datapath
    parser.add_argument("--sst_train", type=str, default="data/ids-sst-train.csv")
    parser.add_argument("--sst_dev", type=str, default="data/ids-sst-dev.csv")
    parser.add_argument("--sst_test", type=str, default="data/ids-sst-test-student.csv")

    parser.add_argument("--para_train", type=str, default="data/quora-train.csv")
    parser.add_argument("--para_dev", type=str, default="data/quora-dev.csv")
    parser.add_argument("--para_test", type=str, default="data/quora-test-student.csv")

    parser.add_argument("--sts_train", type=str, default="data/sts-train.csv")
    parser.add_argument("--sts_dev", type=str, default="data/sts-dev.csv")
    parser.add_argument("--sts_test", type=str, default="data/sts-test-student.csv")

    parser.add_argument("--seed", type=int, default=11711)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--option", type=str,
                        help='pretrain: the BERT parameters are frozen; finetune: BERT parameters are updated',
                        choices=('pretrain', 'finetune'), default="pretrain")
    parser.add_argument("--use_gpu", action='store_true')

    parser.add_argument("--sst_dev_out", type=str, default="predictions/sst-dev-output.csv")
    parser.add_argument("--sst_test_out", type=str, default="predictions/sst-test-output.csv")

    parser.add_argument("--para_dev_out", type=str, default="predictions/para-dev-output.csv")
    parser.add_argument("--para_test_out", type=str, default="predictions/para-test-output.csv")

    parser.add_argument("--sts_dev_out", type=str, default="predictions/sts-dev-output.csv")
    parser.add_argument("--sts_test_out", type=str, default="predictions/sts-test-output.csv")

    # hyper parameters
    parser.add_argument("--batch_size", help='sst: 64, cfimdb: 8 can fit a 12GB GPU', type=int, default=8)
    parser.add_argument("--batch_type", type=str, default="small")
    parser.add_argument("--hidden_dropout_prob", type=float, default=0.3)
    parser.add_argument("--lr", type=float, help="learning rate, default lr for 'pretrain': 1e-3, 'finetune': 1e-5",
                        default=1e-5)
    parser.add_argument("--extension", type=str, default="default")

    # adding the relational layer
    parser.add_argument('--rlayer', type=bool, default=False)

    # adversarial regularization
    parser.add_argument('--pgd_k', type=int, default=1)
    parser.add_argument('--pgd_epsilon', type=float, default=1e-5)
    parser.add_argument('--pgd_lambda', type=float, default=10)

    # bregman momentum
    parser.add_argument('--mbpp_beta', type=float, default=0.995)
    parser.add_argument('--mbpp_mu', type=float, default=1)

    args = parser.parse_args()
    return args