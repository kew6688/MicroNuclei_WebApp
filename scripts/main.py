# Program entry point

from view import View
from model import Model
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Add parent nuclei to mn.')
    parser.add_argument('--model_path', required=True,
                        help='the json file with processed dataset information')
    args = parser.parse_args()
    print(args)

    # model = Model(args.model_path)
    # app = View()
    # app.run()