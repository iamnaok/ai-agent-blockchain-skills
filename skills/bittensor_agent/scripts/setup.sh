#!/bin/bash
# One-time setup for Bittensor Subnet 56 miner

set -e

echo "=== Bittensor Subnet 56 Miner Setup ==="

# Check if btcli is installed
if ! command -v btcli &> /dev/null; then
    echo "Installing btcli..."
    pip install bittensor
fi

# Configuration
read -p "Enter wallet name (default: agent_owner): " WALLET_NAME
WALLET_NAME=${WALLET_NAME:-agent_owner}

read -p "Enter hotkey name (default: miner_01): " HOTKEY_NAME
HOTKEY_NAME=${HOTKEY_NAME:-miner_01}

read -p "Enter subnet UID (default: 56): " NETUID
NETUID=${NETUID:-56}

# Create wallets
echo "Creating coldkey..."
btcli wallet new_coldkey --wallet.name "$WALLET_NAME"

echo "Creating hotkey..."
btcli wallet new_hotkey --wallet.name "$WALLET_NAME" --wallet.hotkey "$HOTKEY_NAME"

# Show addresses
echo "Wallet created!"
echo "Coldkey address: $(btcli wallet list | grep "$WALLET_NAME" | head -1)"

# Fund coldkey
echo ""
echo "=== ACTION REQUIRED ==="
echo "Fund your coldkey with TAO before proceeding"
echo "Address shown above"
read -p "Press Enter once funded..."

# Register on subnet
echo "Registering on subnet $NETUID..."
btcli subnet register --netuid "$NETUID" --wallet.name "$WALLET_NAME" --wallet.hotkey "$HOTKEY_NAME"

# Stake TAO
echo "Staking TAO..."
btcli stake add --netuid "$NETUID" --amount 50 --wallet.name "$WALLET_NAME" --wallet.hotkey "$HOTKEY_NAME"

echo "Setup complete! Run miner with:"
echo "python scripts/miner.py --config config/miner.yaml"
