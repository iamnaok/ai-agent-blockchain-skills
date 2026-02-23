import { 
  ZkProgram, 
  Field, 
  Poseidon,
  Struct,
  SelfProof,
  verify
} from 'o1js';
import { WorkOutput } from '../contracts/AgentReward.js';

/**
 * AgentWorkProgram - ZK circuit for proving AI work
 * 
 * Proves that:
 * 1. A specific AI model was used
 * 2. Specific input was processed
 * 3. Output is correct and unaltered
 * 4. Compute was actually performed
 */

const AgentWorkProgram = ZkProgram({
  name: "AgentWorkProof",
  
  publicInput: WorkOutput,
  publicOutput: Field,

  methods: {
    /**
     * Prove AI work was performed correctly
     * @param publicOutput - The claimed work output
     * @param modelHash - Hash of the AI model weights
     * @param inputHash - Hash of the input data
     * @param actualOutput - Actual output from inference
     * @returns - Verification hash
     */
    proveWork: {
      privateInputs: [Field, Field, Field],
      
      async method(
        publicOutput: WorkOutput,
        modelHash: Field,
        inputHash: Field,
        actualOutput: Field
      ) {
        // In a full ZK-ML implementation, this would:
        // 1. Verify model commitment (hash matches)
        // 2. Prove inference was computed correctly
        // 3. Verify output matches the claimed result
        //
        // For now, we use a simplified hash verification
        // Future: integrate with ZK-ML libraries like EZKL
        
        // Verify output hash matches claimed
        const computedHash = Poseidon.hash([
          modelHash,
          inputHash,
          actualOutput
        ]);
        
        // Assert the result is correct
        publicOutput.resultHash.assertEquals(computedHash);
        
        return computedHash;
      }
    },

    /**
     * Verify a chain of proofs (for iterative tasks)
     */
    verifyChain: {
      privateInputs: [SelfProof<WorkOutput, Field>],
      
      async method(
        publicOutput: WorkOutput,
        previousProof: SelfProof<WorkOutput, Field>
      ) {
        previousProof.verify();
        
        // Aggregate reputation
        const totalCompute = publicOutput.computeUnits.add(
          previousProof.publicInput.computeUnits
        );
        
        return totalCompute;
      }
    }
  }
});

export { AgentWorkProgram };
export default AgentWorkProgram;
