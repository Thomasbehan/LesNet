#!/bin/sh
GREEN='\033[0;32m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

echo -e "${CYAN}Starting the data scraping process...${NC}"
echo -e "${GREEN}Pulling the latest updates from the Git repository...${NC}"
git pull
if [ $? -eq 0 ]; then
    echo -e "${GREEN}Updates pulled successfully.${NC}"
else
    echo -e "${RED}Failed to pull updates. Please check your Git repository.${NC}" >&2
    exit 1
fi

echo -e "${GREEN}Installing required Python packages...${NC}"
python -m pip install -e .[testing]
if [ $? -eq 0 ]; then
    echo -e "${GREEN}Dependencies installed successfully.${NC}"
else
    echo -e "${RED}Failed to install dependencies. Please check your Python environment.${NC}" >&2
    exit 1
fi

echo -e "${GREEN}Running the data scraper script...${NC}"
python commands/run_data_scraper.py
if [ $? -eq 0 ]; then
    echo -e "${CYAN}Data scraping process completed successfully. Check above logs for details.${NC}"
else
    echo -e "${RED}Data scraping process encountered an error. Please check the logs for details.${NC}" >&2
    exit 1
fi

