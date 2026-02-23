# 🤖 AI Agent Blockchain Skills

> Enable your AI agents to earn cryptocurrency by proving their intelligence on-chain.

[![Bittensor](https://img.shields.io/badge/Bittensor-Subnet%2056-blue)](https://subnetalpha.ai/subnet/gradients/)
[![Mina](https://img.shields.io/badge/Mina-zkApps-purple)](https://minaprotocol.com/zkapps)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

---

## 📋 Overview

This repository contains **Agent Zero skills** for two blockchain integration approaches:

| Approach | Network | Mechanism | Best For |
|----------|---------|-----------|----------|
| **Bittensor Agent** | Subnet 56 (Gradients) | Compete on inference quality | Market-based rewards |
| **Mina zkApp Agent** | Mina Protocol | Prove work with ZK-SNARKs | Verifiable privacy |

---

## 🚀 Quick Start

### Bittensor Agent (Market Competition)

```bash
# 1. Install dependencies
pip install bittensor pyyaml

# 2. Setup wallet and register
bash skills/bittensor_agent/scripts/setup.sh

# 3. Run miner
python skills/bittensor_agent/scripts/miner.py --config skills/bittensor_agent/config/miner.yaml
```

### Mina zkApp Agent (Provable Work)

```bash
# 1. Install dependencies
cd skills/mina_zkapp_agent
npm install

# 2. Compile ZK circuits
npm run build

# 3. Deploy contract
zk deploy devnet

# 4. Run agent
npm start
```

---

## 📁 Repository Structure

```
ai-agent-blockchain-skills/
├── skills/
│   ├── bittensor_agent/          # Bittensor Subnet 56 miner
│   │   ├── SKILL.md              # Full documentation
│   │   ├── config/
│   │   │   └── miner.yaml        # Configuration
│   │   └── scripts/
│   │       ├── miner.py          # Miner implementation
│   │       └── setup.sh          # One-time setup
│   │
│   └── mina_zkapp_agent/          # Mina Protocol integration
│       ├── SKILL.md              # Full documentation
│       ├── package.json          # Node.js dependencies
│       └── src/
│           ├── contracts/         # Smart contracts
│           │   └── AgentReward.ts # Reward verification
│           ├── proofs/            # ZK circuits
│           │   └── AgentWork.ts   # Work verification
│           └── agents/            # Agent integration
│               └── MinaAgent.ts   # Full workflow
│
└── README.md                      # This file
```

---

## 🎯 Bittensor Agent

### What It Does

Become a **GOD (Generalized Operator Device)** on Bittensor Subnet 56:

- Receive inference tasks from validators
- Generate AI responses
- Earn TAO based on quality scores
- Build on-chain reputation

### Architecture

```
Validator Query ──▶ Your Agent ──▶ AI Response ──▶ Validator Score ──▶ TAO Reward
```

### Key Metrics

| Metric | Target | Description |
|--------|--------|-------------|
| **Trust** | > 0.8 | Validator confidence |
| **Rank** | Top 20% | Position in leaderboard |
| **Emission** | Variable | TAO per day |

### Requirements

- GPU: NVIDIA A100/H100 or equivalent
- RAM: 32-64GB
- Storage: 500GB+ SSD
- Stake: Variable (check current requirements)

---

## 🔒 Mina zkApp Agent

### What It Does

Prove AI work with **zero-knowledge proofs**:

- Perform inference off-chain (cheap)
- Generate ZK proof (verifiable)
- Submit proof on-chain (~$0.001)
- Earn tokens deterministically

### Architecture

```
AI Work ──▶ ZK Proof ──▶ On-Chain Verification ──▶ Token Mint
   │           │                  │
   ▼           ▼                  ▼
Private    Provable            Verifiable
```

### Key Advantages

| Feature | Benefit |
|---------|---------|
| **Privacy** | Model weights stay private |
| **Verification** | Anyone can verify you did the work |
| **Deterministic** | Reward based on provable compute |
| **Lightweight** | 22KB blockchain |

### Requirements

- MINA tokens: ~1 for fees
- Node.js: v16+
- Any hardware (heavy work off-chain)

---

## 💰 Revenue Models

### Bittensor: Market-Based

```
Your Score = weighted average of validator ratings
Reward = Score^α * Total_Emissions / Sum_Scores

Competition: Other miners on same subnet
Advantage: Better AI = more TAO
```

### Mina: Deterministic

```
Reward = Compute_Units * Rate * Verification

No competition: Your proof is your reward
Advantage: Provable work, not relative performance
```

### Hybrid Approach (Recommended)

Run both:
- **Bittensor** for market validation and TAO earnings
- **Mina** for privacy-critical tasks and ZK verification
- **Bridge reputation** between networks

---

## 🔧 Configuration

### Bittensor (`config/miner.yaml`)

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
```

### Mina (`.env`)

```
CONTRACT_ADDRESS=B62...
AGENT_PRIVATE_KEY=EKE...
NETWORK=devnet  # or mainnet
```

---

## 📊 Monitoring

### Bittensor
- Explorer: https://taostats.io/subnets/56
- Performance: `miner.get_performance()`

### Mina
- Devnet: https://minascan.io/devnet
- Mainnet: https://minascan.io/mainnet
- Reputation: `agent.getReputation()`

---

## 🛠️ Development

### Adding New AI Capabilities

Both skills are designed for easy extension:

**Bittensor:** Edit `_agent_infer()` in `scripts/miner.py`
**Mina:** Edit `runAIInference()` in `src/agents/MinaAgent.ts`

### Testing

```bash
# Bittensor (dry run)
python scripts/miner.py --test

# Mina
npm test
npm run prove
```

---

## 🤝 Contributing

1. Fork the repository
2. Create feature branch: `git checkout -b feature/new-skill`
3. Commit changes: `git commit -am 'Add new skill'`
4. Push to branch: `git push origin feature/new-skill`
5. Open Pull Request

---

## 📚 Resources

### Bittensor
- [Documentation](https://docs.bittensor.com)
- [Subnet 56 Info](https://subnetalpha.ai/subnet/gradients/)
- [Yuma Consensus](https://docs.bittensor.com/yuma-consensus)

### Mina
- [Documentation](https://docs.minaprotocol.com)
- [o1js Guide](https://docs.minaprotocol.com/zkapps/o1js)
- [zkApp Tutorials](https://docs.minaprotocol.com/zkapps/tutorials)

---

## ⚠️ Disclaimer

This is experimental software. Use at your own risk:
- Test on testnets before mainnet
- Manage private keys securely
- Staking involves risk of loss
- Cryptocurrency rewards are volatile

---

## 📝 License

MIT License - See [LICENSE](LICENSE) file

---

*Built with Agent Zero for autonomous AI agents.* 🤖⚡
