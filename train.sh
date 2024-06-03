#!/bin/sh
GREEN='\033[0;32m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

echo -e "${CYAN}Starting the training process for SkinVestigatorAI...${NC}"
echo -e "${GREEN}Pulling latest updates from Git repository...${NC}"
git pull
echo -e "${GREEN}Updates pulled successfully.${NC}"
echo -e "${GREEN}Installing required Python packages...${NC}"
pip install .[testing]
echo -e "${GREEN}Dependencies installed successfully.${NC}"
echo -e "${GREEN}Running training model script...${NC}"
python3 commands/run_train_model.py
echo -e "${CYAN}Training process completed. Check above logs for details.${NC}"
