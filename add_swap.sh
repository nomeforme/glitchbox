#!/bin/bash

# Script to add 64GB of swap space
# Run with sudo: sudo ./add_swap.sh

set -e

SWAP_SIZE="64G"
SWAP_FILE="/swapfile_64g"

echo "This script will create a ${SWAP_SIZE} swap file at ${SWAP_FILE}"
echo "You must run this script with sudo privileges"
echo ""

# Check if running as root
if [ "$EUID" -ne 0 ]; then
    echo "Please run as root (use sudo)"
    exit 1
fi

# Check if swap file already exists
if [ -f "$SWAP_FILE" ]; then
    echo "Swap file already exists at $SWAP_FILE"
    echo "It appears to already be configured. Checking if it's active..."
    if swapon --show | grep -q "$SWAP_FILE"; then
        echo "Swap file is already active!"
        swapon --show
        exit 0
    else
        echo "Swap file exists but is not active. Activating it..."
        swapon "$SWAP_FILE"
        echo "Done!"
        swapon --show
        exit 0
    fi
fi

echo "Creating ${SWAP_SIZE} swap file..."
echo "This may take several minutes..."

# Create swap file using fallocate (faster) or dd (fallback)
if command -v fallocate &> /dev/null; then
    fallocate -l 64G "$SWAP_FILE"
else
    dd if=/dev/zero of="$SWAP_FILE" bs=1M count=65536 status=progress
fi

echo "Setting permissions..."
chmod 600 "$SWAP_FILE"

echo "Making swap file..."
mkswap "$SWAP_FILE"

echo "Enabling swap..."
swapon "$SWAP_FILE"

echo "Verifying swap is active..."
swapon --show
free -h

# Ask if user wants to make it permanent
echo ""
read -p "Do you want to make this swap permanent (add to /etc/fstab)? (y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    # Check if entry already exists in fstab
    if grep -q "$SWAP_FILE" /etc/fstab; then
        echo "Entry already exists in /etc/fstab"
    else
        echo "Adding entry to /etc/fstab..."
        echo "$SWAP_FILE none swap sw 0 0" >> /etc/fstab
        echo "Swap will now persist across reboots"
    fi
fi

echo ""
echo "Done! 64GB swap has been added successfully."
echo "Current swap status:"
free -h | grep -i swap
