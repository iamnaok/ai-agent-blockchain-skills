# Bittensor Agent Skill

## Metadata
- **Name**: bittensor_agent
- **Version**: 1.0.0
- **Author**: AgentZero
- **Tags**: blockchain, ai, bittensor, mining, subnet56
- **Description**: Enable Agent Zero to participate as a miner on Bittensor Subnet 56 (Gradients) - the AI Agent Layer

## Overview

This skill enables your AI agent to become a **GOD (Generalized Operator Device)** on Bittensor Subnet 56, earning TAO tokens for providing AI inference services.

### What is a GOD?
> Generalized Operator Devices are persistent AI agents that respond to users, store memory, and evolve over time. They run as programmable services accessible via API and generate real TAO revenue.

## Architecture

```
Agent Zero ──▶ GOD Service ──▶ Bittensor Subnet 56 ──▶ TAO Rewards
     │              │                  │
     ▼              ▼                  ▼
Task Routing   Inference API    Yuma Consensus
Memory Sync    Scoring          Emissions
```

## Prerequisites

| Requirement | Details |
|-------------|---------|
| **Hardware** | GPU (NVIDIA A100/H100 or equivalent) |
| **RAM** | 32-64GB minimum |
| **Storage** | 500GB+ SSD |
| **Stake** | Varies by subnet (check current requirements) |
| **TAO** | For staking and transaction fees |

## Installation

```bash
# Install Bittensor SDK
pip install bittensor

# Create wallet
btcli wallet new_coldkey --wallet.name agent_owner
btcli wallet new_hotkey --wallet.name agent_owner --wallet.hotkey miner_01

# Register on Subnet 56
btcli subnet register --netuid 56 --wallet.name agent_owner --wallet.hotkey miner_01

# Stake TAO
btcli stake add --netuid 56 --amount 50 --wallet.name agent_owner --wallet.hotkey miner_01
```

## Methods

### `query(task: str, context: dict) -> str`
Send a task to the GOD service and return the AI-generated response.

**Parameters:**
- `task` (str): The task description or query
- `context` (dict, optional): Previous conversation context

**Returns:**
- `response` (str): Generated response scored by validators

### `get_performance() -> dict`
Get current performance metrics from the metagraph.

**Returns:**
```python
{
    "rank": float,        # Position in miner leaderboard
    "trust": float,       # Validator confidence (target > 0.8)
    "consensus": float,   # Agreement score
    "incentive": float,   # Current incentive weight
    "emission_per_day": float  # TAO earned per day
}
```

### `sync_memory()`
Sync agent memory with on-chain persistence layer.

## Usage Example

```python
from skills.bittensor_agent import BittensorAgentSkill

# Initialize agent
agent = BittensorAgentSkill()

# Receive task from validator
response = await agent.query(
    task="Explain zero-knowledge proofs",
    context={"user_id": "123", "session": "abc"}
)

# Check performance
metrics = agent.get_performance()
print(f"Daily TAO earnings: {metrics['emission_per_day']}")
```

## Reward Mechanism

TAO emissions are determined by **Yuma Consensus**:

1. **Validators** send queries to miners
2. **Miners** (your agent) process and respond
3. **Validators** score responses on quality, speed, consistency
4. **Emissions** distributed proportional to performance ranking

**Your Score** = weighted average across all validators
**Reward** ∝ Score^α (α varies by subnet)

## Configuration

Edit `config/miner.yaml`:

```yaml
wallet:
  name: agent_owner
  hotkey: miner_01

subnet:
  netuid: 56
  
agent:
  max_tokens: 2048
  temperature: 0.7
  memory_enabled: true
  
logging:
  level: INFO
  file: logs/miner.log
```

## Scripts

### `scripts/miner.py`
Continuous miner process. Run with:
```bash
python scripts/miner.py --config config/miner.yaml
```

### `scripts/setup.sh`
One-time setup script for wallet creation and registration.

## Monitoring

Track your miner:
- **Explorer**: https://taostats.io/subnets/56
- **API**: Use `get_performance()` method
- **Logs**: Check `logs/miner.log`

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Low trust score | Improve response quality and consistency |
| Not receiving queries | Check stake amount and registration |
| High latency | Optimize inference pipeline |
| Connection errors | Verify network and subtensor sync |

## References

- [Bittensor Docs](https://docs.bittensor.com)
- [Subnet 56 Info](https://subnetalpha.ai/subnet/gradients/)
- [Gradients Deep Dive](https://asymmetricjump.substack.com/p/gradients-subnet-56-the-ai-agent)

## License

MIT - See LICENSE file
