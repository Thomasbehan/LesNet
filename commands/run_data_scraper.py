import argparse
from skinvestigatorai.services.data_scaper import DataScraper


def main():
    parser = argparse.ArgumentParser(
        description="Download images from ISIC Archive and split into training and testing sets.")
    parser.add_argument("-p", "--pages", type=int, default=-1,
                        help="Number of pages to download. Default is -1, which downloads all pages.")
    args = parser.parse_args()

    scraper = DataScraper(max_pages=args.pages)
    scraper.download_images()
    scraper.balance_dataset_and_save()


if __name__ == "__main__":
    main()
