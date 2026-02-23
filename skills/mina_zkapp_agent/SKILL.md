# Mina zkApp Agent Skill

## Metadata
- **Name**: mina_zkapp_agent
- **Version**: 1.0.0
- **Author**: AgentZero
- **Tags**: blockchain, zkp, mina, ai, privacy, verification
- **Description**: Enable Agent Zero to prove AI work via zk-SNARKs on Mina Protocol and earn verified rewards

## Overview

This skill enables your AI agent to:
1. Perform AI inference **off-chain** (where compute is cheap)
2. Generate **ZK proofs** that the work was done correctly
3. Submit proofs **on-chain** for verification
4. Earn tokens based on **provable** work, not promises

### Why Mina?
- **22KB blockchain** — Always lightweight
- **Native ZK** — Privacy-preserving by design
- **Off-chain compute** — Heavy AI runs client-side
- **Provable AI** — Verify inference without revealing models

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                         Agent Zero                            │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐  │
│  │   User Task  │───▶│  Mina Agent  │───▶│  ZK Prover   │  │
│  │   Request    │    │   (wrapper)  │    │  (generate)  │  │
│  └──────────────┘    └──────────────┘    └──────┬───────┘  │
└──────────────────────────────────────────────────┼──────────┘
                                                   │
                    ┌──────────────────────────────┼────────────────┐
                    │                              ▼                 │
                    │  OFF-CHAIN              ON-CHAIN (Mina)      │
                    │  ┌─────────────┐         ┌─────────────┐      │
                    │  │  AI Model   │────────▶│  zkApp      │      │
                    │  │  Inference  │  Proof  │  Contract   │      │
                    │  │  (Private)  │         │  (Verify)   │      │
                    │  └─────────────┘         └──────┬──────┘      │
                    │         │                        │             │
                    │         ▼                        ▼             │
                    │  ┌─────────────┐         ┌─────────────┐      │
                    │  │  ZK Proof   │         │  Token Mint │      │
                    │  │  (o1js)     │         │  (Verified) │      │
                    │  └─────────────┘         └─────────────┘      │
                    └─────────────────────────────────────────────────┘
```

## Prerequisites

| Requirement | Details |
|-------------|---------|
| **MINA** | For transaction fees (~1 MINA minimum) |
| **Node.js** | v16+ for o1js |
| **TypeScript** | For zkApp development |
| **Compute** | Any machine (heavy work off-chain) |

## Installation

```bash
# Install zkApp CLI
npm install -g zkapp-cli

# Initialize project
zkapp-cli create mina-zkapp-agent
cd mina-zkapp-agent

# Install dependencies
npm install
npm install o1js@latest

# Fund account (devnet)
# https://faucet.minaprotocol.com/
```

## Smart Contract: AgentReward

Verifies ZK proofs and mints rewards.

```typescript
// src/contracts/AgentReward.ts

export class AgentReward extends SmartContract {
  @state(Field) totalMinted = State<Field>();
  @state(Field) agentReputation = State<Field>();
  @state(Field) rewardRate = State<Field>();

  @method
  async verifyAndMint(workProof: WorkProof, publicKey: PublicKey) {
    // Verify ZK proof (magic happens here)
    workProof.verify();
    
    // Calculate reward
    const reward = output.computeUnits.mul(rate);
    
    // Update reputation
    this.agentReputation.set(currentRep.add(output.computeUnits));
    
    // Emit mint event
    this.emitEvent('Mint', {
      agent: publicKey,
      amount: reward,
      task: output.taskId
    });
  }
}
```

## ZK Program: AgentWork

Generates proofs that AI work was done correctly.

```typescript
// src/proofs/AgentWork.ts

const AgentWorkProgram = ZkProgram({
  name: "AgentWorkProof",
  
  publicInput: WorkOutput,
  publicOutput: Field,

  methods: {
    proveWork: {
      privateInputs: [Field, Field, Field],
      
      async method(output, modelHash, inputHash, actualOutput) {
        // Verify:
        // 1. Model hash matches committed weights
        // 2. Input was processed
        // 3. Output is correct
        
        const computedHash = Poseidon.hash([
          modelHash, inputHash, actualOutput
        ]);
        
        output.resultHash.assertEquals(computedHash);
        
        return computedHash;
      }
    }
  }
});
```

## Methods

### `executeTask(task: string, context: any): Promise<Result>`

Complete workflow: Run AI → Generate proof → Submit → Earn reward.

**Parameters:**
- `task` (string): The AI task
- `context` (object): Context data

**Returns:**
```typescript
{
  result: any,           // AI output
  proof: string,         // ZK proof
  txHash: string,        // Mina transaction
  reward?: number        // Tokens earned
}
```

### `generateWorkProof(task, result, computeUnits): Promise<Proof>`

Generates ZK proof for completed work.

**Parameters:**
- `task` (string): Original task
- `result` (any): AI output
- `computeUnits` (number): Computational cost

**Returns:**
- ZK proof ready for on-chain verification

### `getReputation(): Promise<number>`

Get accumulated reputation from verified work.

## Usage Example

```typescript
import { MinaAgentSkill } from './src/agents/MinaAgent';

// Initialize
const agent = new MinaAgentSkill(
  contractAddress: "B62...",
  agentPrivateKey: "EKE..."
);

// Execute task (full workflow)
const result = await agent.executeTask(
  "Analyze this smart contract",
  { contractCode: "..." }
);

console.log("TX Hash:", result.txHash);
console.log("Proof verified on-chain!");

// Get reputation
const reputation = await agent.getReputation();
console.log("Reputation:", reputation);
```

## Resource-to-Reward Mapping

| Resource Spent | Proof Element | Reward Formula |
|----------------|---------------|----------------|
| Compute time | Timestamp + complexity | `computeUnits * rate` |
| Model inference | Model hash | Verified in circuit |
| Data processed | Input hash | Privacy preserved |
| Output quality | Result hash | Verifiable |

## Files

| File | Purpose |
|------|---------|
| `src/contracts/AgentReward.ts` | Smart contract |
| `src/proofs/AgentWork.ts` | ZK circuit |
| `src/agents/MinaAgent.ts` | Agent integration |
| `package.json` | Dependencies |

## Deployment

```bash
# Compile circuits
npm run build

# Deploy to devnet
zk deploy devnet

# Deploy to mainnet
zk deploy mainnet
```

## Testing

```bash
# Run local tests
npm test

# Test proof generation
node scripts/test-proof.js
```

## Monitoring

- **Devnet Explorer**: https://minascan.io/devnet
- **Mainnet Explorer**: https://minascan.io/mainnet
- **API**: Use `getReputation()` method

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Proof too large | Optimize circuit, reduce constraints |
| Verification fails | Check model/input hashes |
| Low balance | Fund account from faucet |
| Slow proving | Use more powerful hardware |

## References

- [Mina Docs](https://docs.minaprotocol.com)
- [o1js Docs](https://docs.minaprotocol.com/zkapps/o1js)
- [zkApp Tutorials](https://docs.minaprotocol.com/zkapps/tutorials)

## License

MIT - See LICENSE file
