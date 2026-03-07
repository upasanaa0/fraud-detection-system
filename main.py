import argparse
import time


def parse_args():
    parser = argparse.ArgumentParser(description="Fraud Detection Pipeline")
    parser.add_argument("--step", choices=["generate", "train", "predict", "all"], default="all")
    parser.add_argument("--train-rows", type=int, default=500000)
    parser.add_argument("--prod-rows",  type=int, default=100000)
    parser.add_argument("--data-dir",   default="data")
    parser.add_argument("--model-dir",  default="model")
    return parser.parse_args()


def run_generate(args):
    from src.data_generator import create_datasets
    print("\nSTEP 1 - DATA GENERATION")
    print("=" * 50)
    t = time.time()
    create_datasets(
        train_rows=args.train_rows,
        prod_rows=args.prod_rows,
        output_dir=args.data_dir
    )
    print(f"Done in {time.time()-t:.1f}s")


def run_train(args):
    from src.train import train
    print("\nSTEP 2 - MODEL TRAINING")
    print("=" * 50)
    t = time.time()
    train(
        data_path=f"{args.data_dir}/train.csv",
        model_dir=args.model_dir
    )
    print(f"Done in {time.time()-t:.1f}s")


def run_predict(args):
    from src.predict import predict
    print("\nSTEP 3 - INFERENCE AND SCORING")
    print("=" * 50)
    t = time.time()
    predict(
        data_path=f"{args.data_dir}/production.csv",
        model_dir=args.model_dir,
        output_path=f"{args.data_dir}/final_output.csv"
    )
    print(f"Done in {time.time()-t:.1f}s")


def main():
    args = parse_args()
    total_start = time.time()

    if args.step in ("all", "generate"):
        run_generate(args)
    if args.step in ("all", "train"):
        run_train(args)
    if args.step in ("all", "predict"):
        run_predict(args)

    print(f"\nPipeline complete - total time: {time.time()-total_start:.1f}s")
    print(f"Output: {args.data_dir}/final_output.csv")


if __name__ == "__main__":
    main()