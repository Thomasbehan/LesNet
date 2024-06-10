from skinvestigatorai.services.model import SVModel

def main():
    model_service = SVModel()
    model, _ = model_service.load_model()
    model_service.quantize_model(model)
    print("Model Quantized")


if __name__ == "__main__":
    main()
