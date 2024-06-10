from skinvestigatorai.services.data_scaper import DataScraper


def main():
    scraper = DataScraper()
    scraper.balance_dataset_and_save()


if __name__ == "__main__":
    main()
