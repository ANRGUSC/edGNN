from tempfile import TemporaryDirectory
from edgnn.core.data.dglrgcn import preprocess_dglrgcn
from gcnsched.run_model import run_model
import pathlib 

thisdir = pathlib.Path(__file__).resolve().parent

def main():
    with TemporaryDirectory() as tmp:
        path = pathlib.Path(tmp)
        preprocessed_graphs = path.joinpath("preprocessed_graphs", "aifb")
        preprocessed_graphs.mkdir(exist_ok=True, parents=True)
        preprocess_dglrgcn(
            dataset="aifb",
            out_folder=str(preprocessed_graphs),
            reverse_edges=True
        )

        config_path = thisdir.joinpath("config_files", "config_edGNN_node_class.json")
        run_model(
            arg_dataset="aifb",
            arg_config_fpath=str(config_path),
            arg_data_path=str(preprocessed_graphs),
            arg_lr=0.005,
            arg_n_epochs=400,
            arg_weight_decay=0
        )


if __name__ == "__main__":
    main()