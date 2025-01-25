#!/bin/bash

# Define the target directory
TARGET_DIR="data/train"

# Use find to gather all image files into an array
mapfile -t images < <(find "$TARGET_DIR" -type f \( -iname "*.jpg" -o -iname "*.jpeg" -o -iname "*.png" -o -iname "*.gif" \))

# Count total number of images
total_images=${#images[@]}

# Initialize counter
counter=0

# Function to process images
process_image() {
    local img="$1"
    convert "$img" -resize x512 -quality 85 "$img"
}

# Export the function for parallel execution
export -f process_image

# Use GNU Parallel to speed up processing
printf "\nProcessing: [%-50s] %d%%" "$(printf ' %.0s' {1..50})" 0
printf "\n"

# Process images in parallel and update progress
{
    for img in "${images[@]}"; do
        process_image "$img" &
        counter=$((counter + 1))
        percent=$((counter * 100 / total_images))

        # Display progress bar (update every 5 images to reduce output)
        if (( counter % 5 == 0 )); then
            printf "\rProcessing: [%-50s] %d%%" "$(printf '#%.0s' $(seq 1 $((percent / 2))))$(printf ' %.0s' $(seq $((percent / 2 + 1)) 50))" "$percent"
        fi
    done

    wait # Wait for all background processes to finish
}

echo -e "\nImage compression and resizing completed."
