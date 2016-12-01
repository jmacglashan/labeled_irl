package burlap.behavior.stochasticgames.auxiliary.jointmdp;


import burlap.mdp.core.action.Action;
import burlap.mdp.core.state.State;
import burlap.mdp.singleagent.model.RewardFunction;
import burlap.mdp.stochasticgames.JointAction;
import burlap.mdp.stochasticgames.model.JointRewardFunction;

import java.util.Map;

/**
 * @author James MacGlashan.
 */
public class TotalWelfare implements RewardFunction {

	JointRewardFunction jr;

	public TotalWelfare(JointRewardFunction jr) {
		this.jr = jr;
	}

	@Override
	public double reward(State s, Action a, State sprime) {

		double sum = 0.;
		double [] rs = this.jr.reward(s, (JointAction)a, sprime);
		for(double d : rs){
			sum += d;
		}
		return sum;
	}
}
