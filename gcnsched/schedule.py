import torch
import networkx as nx
import pathlib
from edgnn.core.app import App

thisdir = pathlib.Path(__file__).resolve().parent

def schedule():
    app = App()
    app.model.load_state_dict(torch.load(thisdir.joinpath("model")))
    print(app.model)


def main():
    schedule()

if __name__ == "__main__":
    main()
