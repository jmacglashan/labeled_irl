package burlap.behavior.stochasticgames.auxiliary.jointmdp;

import burlap.behavior.policy.EnumerablePolicy;
import burlap.behavior.policy.Policy;
import burlap.behavior.policy.support.ActionProb;
import burlap.mdp.core.action.Action;
import burlap.mdp.core.state.State;
import burlap.mdp.stochasticgames.JointAction;


import java.util.List;

/**
 * @author James MacGlashan.
 */
public class DecentralizedPolicy implements Policy{

	protected Policy cpolicy;
	protected int agentId;


	public DecentralizedPolicy(Policy cpolicy) {
		this.cpolicy = cpolicy;
	}

	public DecentralizedPolicy(Policy cpolicy, int agentId) {
		this.cpolicy = cpolicy;
		this.agentId = agentId;
	}

	public Policy getCpolicy() {
		return cpolicy;
	}

	public void setCpolicy(Policy cpolicy) {
		this.cpolicy = cpolicy;
	}

	public int getAgentId() {
		return agentId;
	}

	public void setAgentId(int agentId) {
		this.agentId = agentId;
	}


    @Override
    public Action action(State s) {
        JointAction ja = (JointAction)cpolicy.action(s);
        Action myAction = ja.action(this.agentId);
        return myAction;
    }

    @Override
    public double actionProb(State s, Action a) {
        double sum = 0.;
        List<ActionProb> aps = ((EnumerablePolicy)this.cpolicy).policyDistribution(s);
        for(ActionProb ap : aps){
            JointAction ja = (JointAction)ap.ga;
            if(ja.action(this.agentId).equals(a)){
                sum += ap.pSelection;
            }
        }
        return sum;
    }

    @Override
    public boolean definedFor(State s) {
        return this.cpolicy.definedFor(s);
    }

}
