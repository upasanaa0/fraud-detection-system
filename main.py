import argparse, time


def parse_args():
    p = argparse.ArgumentParser(description="Fraud Detection Pipeline")
    p.add_argument("--step", choices=["generate", "train", "predict", "evaluate", "all"], default="all")
    p.add_argument("--train-rows", type=int, default=500000)
    p.add_argument("--prod-rows", type=int, default=100000)
    p.add_argument("--data-dir", default="data")
    p.add_argument("--model-dir", default="model")
    return p.parse_args()


def main():
    a, t0 = parse_args(), time.time()
    if a.step in ("all", "generate"):
        from src.data_generator import create_datasets
        create_datasets(a.train_rows, a.prod_rows, a.data_dir)
    if a.step in ("all", "train"):
        from src.train import train
        train(f"{a.data_dir}/train.csv", a.model_dir)
    if a.step in ("all", "predict"):
        from src.predict import predict
        predict(f"{a.data_dir}/production.csv", a.model_dir, f"{a.data_dir}/final_output.csv", f"{a.data_dir}/train.csv")
    if a.step in ("all", "evaluate"):
        from src.evaluate import evaluate
        evaluate(a.model_dir, a.data_dir)
    print(f"done in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()