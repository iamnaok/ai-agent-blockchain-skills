import { 
  SmartContract, 
  state, 
  State, 
  method, 
  Proof,
  Field,
  PublicKey,
  Struct,
  Poseidon,
  UInt64
} from 'o1js';

// Define what constitutes "useful work"
export class WorkOutput extends Struct({
  taskId: Field,
  resultHash: Field,
  computeUnits: Field,
  timestamp: Field
}) {}

export class WorkProof extends Proof<WorkOutput, Field> {}

/**
 * AgentReward - Smart contract for verifying AI work and minting rewards
 * 
 * Agents perform AI inference off-chain, generate ZK proofs,
 * and submit for verification. Valid proofs earn tokens.
 */
export class AgentReward extends SmartContract {
  // Token supply tracking
  @state(Field) totalMinted = State<Field>();
  
  // Agent reputation (accumulated valid work)
  @state(Field) agentReputation = State<Field>();
  
  // Reward rate per compute unit (adjustable)
  @state(Field) rewardRate = State<Field>();

  events = {
    'Mint': Struct({
      agent: PublicKey,
      amount: Field,
      task: Field
    })
  };

  init() {
    super.init();
    this.totalMinted.set(Field(0));
    this.agentReputation.set(Field(0));
    this.rewardRate.set(Field(1000)); // 1000 tokens per compute unit
  }

  /**
   * Verify ZK proof and mint reward
   * @param workProof - ZK proof of AI work
   * @param publicKey - Agent's public key
   */
  @method
  async verifyAndMint(
    workProof: WorkProof,
    publicKey: PublicKey
  ) {
    // Verify the ZK proof (this is the magic)
    workProof.verify();
    
    const output = workProof.publicInput;
    
    // Calculate reward based on compute units
    const rate = await this.rewardRate.get();
    const reward = output.computeUnits.mul(rate);
    
    // Update agent reputation
    const currentRep = await this.agentReputation.get();
    this.agentReputation.set(currentRep.add(output.computeUnits));
    
    // Update total minted
    const currentSupply = await this.totalMinted.get();
    this.totalMinted.set(currentSupply.add(reward));
    
    // Emit event for off-chain indexing
    this.emitEvent('Mint', {
      agent: publicKey,
      amount: reward,
      task: output.taskId
    });
  }

  /**
   * Get agent's reputation
   */
  async getReputation(agent: PublicKey): Promise<Field> {
    return await this.agentReputation.get();
  }

  /**
   * Admin: Update reward rate
   */
  @method
  async setRewardRate(newRate: Field) {
    // Add admin check here in production
    this.rewardRate.set(newRate);
  }
}

export default AgentReward;
