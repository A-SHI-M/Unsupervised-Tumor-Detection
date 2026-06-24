from tumor_detection import logger
from tumor_detection.config.configuration import ConfigurationManager
from tumor_detection.components.model_evaluation import ModelEvaluator

STAGE_NAME = "Model Evaluation Stage"


class ModelEvaluationPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        model_evaluation_config = config.get_model_evaluation_config()
        evaluator = ModelEvaluator(config=model_evaluation_config)
        evaluator.evaluate()


if __name__ == "__main__":
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        pipeline = ModelEvaluationPipeline()
        pipeline.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e
