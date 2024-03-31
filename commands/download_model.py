import argparse
from skinvestigatorai.models.downloader import downloader


def main():
    parser = argparse.ArgumentParser(description="Download a specific AI model.")
    parser.add_argument("-m", "--modelname", required=True, help="The name of the model to download.")

    args = parser.parse_args()

    model_name = args.modelname
    if downloader(model_name):
        print(f"Successfully downloaded the model: {model_name}")
    else:
        print(f"Failed to download the model: {model_name}")


if __name__ == "__main__":
    main()
