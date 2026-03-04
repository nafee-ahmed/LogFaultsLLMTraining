from datasets import load_dataset, concatenate_datasets

if __name__ == "__main__":
    # load dataset from csv
    precomp_ds = load_dataset(
        "csv", data_files="./datasets/precomputed/dirty_datasetv3/train.csv"
    )["train"]
    orig_ds = load_dataset(
        "csv", data_files="./datasets/prepared/dirty_datasetv1/train.csv"
    )["train"]

    # combine datasets
    combined_ds = concatenate_datasets([precomp_ds, orig_ds]).shuffle(seed=42)

    # save combined dataset to csv
    combined_ds.to_csv("./datasets/combined/dirty_datasetv3.csv", index=False)
