# Program entry point

from view import View
from model import Model
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Add parent nuclei to mn.')
    parser.add_argument('mn_model_path', help='the path to pre trained model for mn segment')
    parser.add_argument('nuc_model_path', help='the path to pre trained model for nuc segment')
    args = parser.parse_args()

    model = Model(args.mn_model_path, args.nuc_model_path)
    app = View(model)
    app.run()