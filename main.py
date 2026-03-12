from src.pipeline.trainer_pipeline import TrainPipeline

if __name__ == "__main__":

    print("Starting Training Pipeline...")

    pipeline = TrainPipeline()

    pipeline.run_pipeline() # type: ignore

    print("Training Completed")