from argparse import ArgumentParser

def demo_get_args():
    parser = ArgumentParser(description='Arguments for demo')

    parser.add_argument("--use_gpu", action='store_true')


    parser.add_argument('--sentiment_anal', action='store_true')
    parser.add_argument('--paraphrase_detect', action='store_true')
    parser.add_argument('--semantic_sim', action='store_true')


    args = parser.parse_args()
    return args