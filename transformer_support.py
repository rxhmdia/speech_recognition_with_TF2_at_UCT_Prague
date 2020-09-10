from transformers import pipeline


def masked_pipeline_from_trained_model(path_to_model_folder):
    return pipeline("fill-mask",
                    model=path_to_model_folder,
                    tokenizer=path_to_model_folder)