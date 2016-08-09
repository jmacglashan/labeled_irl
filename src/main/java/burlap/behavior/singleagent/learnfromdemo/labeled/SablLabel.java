package burlap.behavior.singleagent.learnfromdemo.labeled;

import burlap.behavior.functionapproximation.FunctionGradient;
import burlap.behavior.policy.BoltzmannQPolicy;
import burlap.behavior.policy.Policy;
import burlap.behavior.singleagent.Episode;
import burlap.behavior.singleagent.learnfromdemo.CustomRewardModel;
import burlap.behavior.singleagent.learnfromdemo.mlirl.MLIRLRequest;
import burlap.behavior.singleagent.learnfromdemo.mlirl.support.BoltzmannPolicyGradient;
import burlap.behavior.singleagent.learnfromdemo.mlirl.support.DifferentiableQFunction;
import burlap.behavior.singleagent.learnfromdemo.mlirl.support.DifferentiableRF;
import burlap.behavior.valuefunction.QProvider;
import burlap.datastructures.HashedAggregator;
import burlap.debugtools.DPrint;
import burlap.mdp.core.action.Action;
import burlap.mdp.core.state.State;

import java.util.List;
import java.util.Map;

/**
 * @author James MacGlashan.
 */
public class SablLabel {

	/**
	 * The MLRIL request defining the IRL problem.
	 */
	protected MLIRLRequest request;

	/**
	 * The gradient ascent learning rate
	 */
	protected double learningRate;

	/**
	 * The likelihood change threshold to stop gradient ascent.
	 */
	protected double maxLikelihoodChange;

	/**
	 * The maximum number of steps of gradient ascent. when set to -1, there is no limit and termination will be
	 * based on the {@link #maxLikelihoodChange} alone.
	 */
	protected int maxSteps;



	protected int maxFeedbackSamples = 50;

	protected double phi = 1.;


	/**
	 * Initializes.
	 * @param request the problem request definition
	 * @param learningRate the gradient ascent learning rate
	 * @param maxLikelihoodChange the likelihood change threshold that must be reached to terminate gradient ascent
	 * @param maxSteps the maximum number of gradient ascent steps allowed before termination is forced. Set to -1 to rely only on likelihood threshold.
	 */
	public SablLabel(MLIRLRequest request, double learningRate, double maxLikelihoodChange, int maxSteps){

		this.request = request;
		this.learningRate = learningRate;
		this.maxLikelihoodChange = maxLikelihoodChange;
		this.maxSteps = maxSteps;

		if(!request.isValid()){
			throw new RuntimeException("Provided MLIRLRequest object is not valid.");
		}

	}

	/**
	 * The debug code used for printing information to the terminal.
	 */
	protected int debugCode = 625420;

	/**
	 * Sets the {@link burlap.behavior.singleagent.learnfromdemo.mlirl.MLIRLRequest} object defining the IRL problem.
	 * @param request the {@link burlap.behavior.singleagent.learnfromdemo.mlirl.MLIRLRequest} object defining the IRL problem.
	 */
	public void setRequest(LabeledIRLRequest request){
		this.request = request;
	}


	/**
	 * Sets whether information during learning is printed to the terminal. Will automatically toggle the debug printing
	 * for the underlying valueFunction as well.
	 * @param printDebug if true, information is printed to the terminal; if false then it is silent.
	 */
	public void toggleDebugPrinting(boolean printDebug){
		DPrint.toggleCode(this.debugCode, printDebug);
		this.request.getPlanner().toggleDebugPrinting(printDebug);
	}


	/**
	 * Returns the debug code used for printing to the terminal
	 * @return the debug code used for printing to the terminal.
	 */
	public int getDebugCode(){
		return this.debugCode;
	}


	/**
	 * Sets the debug code used for printing to the terminal
	 * @param debugCode the debug code used for printing to the terminal
	 */
	public void setDebugCode(int debugCode){
		this.debugCode = debugCode;
	}

	/**
	 * Runs gradient ascent.
	 */
	public void performIRL(){

		DifferentiableRF rf = this.request.getRf();

		//reset valueFunction
		this.request.getPlanner().resetSolver();
		this.request.getPlanner().setModel(new CustomRewardModel(request.getDomain().getModel(), rf));
		double lastLikelihood = this.logLikelihood();
		DPrint.cl(this.debugCode, "RF: " + this.request.getRf().toString());
		DPrint.cl(this.debugCode, "Log likelihood: " + lastLikelihood);


		int i;
		for(i = 0; i < maxSteps || this.maxSteps == -1; i++){

			//move up gradient
			FunctionGradient gradient = this.gradientLogLikelihood();
			double maxChange = 0.;
			for(FunctionGradient.PartialDerivative pd : gradient.getNonZeroPartialDerivatives()){
				double curVal = rf.getParameter(pd.parameterId);
				double nexVal = curVal + this.learningRate*pd.value;
				rf.setParameter(pd.parameterId, nexVal);
				double delta = Math.abs(curVal-nexVal);
				maxChange = Math.max(maxChange, delta);
			}


			//reset valueFunction
			this.request.getPlanner().resetSolver();
			this.request.getPlanner().setModel(new CustomRewardModel(request.getDomain().getModel(), rf));

			double newLikelihood = this.logLikelihood();
			double likelihoodChange = newLikelihood-lastLikelihood;
			lastLikelihood = newLikelihood;


			DPrint.cl(this.debugCode, "RF: " + this.request.getRf().toString());
			DPrint.cl(this.debugCode, "Log likelihood: " + lastLikelihood + " (change: " + likelihoodChange + ")");

			if(Math.abs(likelihoodChange) < this.maxLikelihoodChange){
				i++;
				break;
			}


		}


		DPrint.cl(this.debugCode, "\nNum gradient ascent steps: " + i);
		DPrint.cl(this.debugCode, "RF: " + this.request.getRf().toString());



	}



	protected FunctionGradient gradientLogLikelihood(){
		HashedAggregator<Integer> agg = new HashedAggregator<>();
		List<Episode> demos = this.request.getExpertEpisodes();
		for(int i = 0; i < demos.size(); i++){
			FunctionGradient fg = this.gradientLogLikelihood(demos.get(i));
			for(FunctionGradient.PartialDerivative pd : fg.getNonZeroPartialDerivatives()){
				agg.add(pd.parameterId, pd.value);
			}
		}

		FunctionGradient grad = new FunctionGradient.SparseGradient(agg.size());
		for(Map.Entry<Integer, Double> e : agg.entrySet()){
			grad.put(e.getKey(), e.getValue());
		}

		return grad;

	}


	protected double logLikelihood(){

		double sum = 0.;
		List<Episode> demos = this.request.getExpertEpisodes();
		for(int i = 0; i < demos.size(); i++){
			double val = this.logLikelihood(demos.get(i));
			sum += val;
		}

		return sum;

	}



	protected FunctionGradient gradientLogLikelihood(Episode e){
		return this.gradEpisodeFeedback(e);
	}


	/**
	 * Produces log likelihood for an episode with potentially missing feedback labels. Feedbacks are specified with
	 * the rewards in the episode. A zero reward indicates a missing feedback label. Otherwise should be +1 or -1.
	 * Multiple samples of different feedback assignments will be generated and used to estimate the likelihood
	 * @param e the episode
	 * @return the log likelihood.
	 */
	protected double logLikelihood(Episode e){
		return logSumProbFeedback(e);
	}





	/**
	 * gradient for some episode of feedback assignments. Equals gradient of gxi term.
	 * @param episode the episode with feedback assignments in the rewards
	 * @return the gradient
	 */
	protected FunctionGradient gradEpisodeFeedback(Episode episode){

		Policy p = new BoltzmannQPolicy((QProvider)this.request.getPlanner(), 1./this.request.getBoltzmannBeta());
		HashedAggregator <Integer> gradSum = new HashedAggregator<>();
		for(int t = 1; t < episode.numTimeSteps(); t++){
			double f = episode.reward(t);
			State s = episode.state(t-1);
			Action a = episode.action(t-1);
			FunctionGradient grad = this.gradProbFeedback(f, s, a);
			for(FunctionGradient.PartialDerivative pd : grad.getNonZeroPartialDerivatives()){
				double pxi = this.probFeedback(f, s, a, p);
				double div = pd.value / pxi;
				gradSum.add(pd.parameterId, div);
			}
		}

		FunctionGradient fg = new FunctionGradient.SparseGradient(gradSum.size());
		for(Map.Entry<Integer, Double> e : gradSum.entrySet()){
			fg.put(e.getKey(), e.getValue());
		}

		return fg;

	}


	protected double logSumProbFeedback(Episode e){

		Policy p = new BoltzmannQPolicy((QProvider)this.request.getPlanner(), 1./this.request.getBoltzmannBeta());

		double sum = 0.;
		for(int t = 1; t < e.numTimeSteps(); t++){
			double pr = this.probFeedback(e.reward(t), e.state(t-1), e.action(t-1), p);
			double lg = Math.log(pr);
			sum += lg;
		}
		return sum;

	}





	/**
	 * Gradient of a feedback. Equals gradient of P_xij in doc.
	 * @param feedback the feedback value
	 * @param s state
	 * @param a action
	 * @return the gradient
	 */
	protected FunctionGradient gradProbFeedback(double feedback, State s, Action a){
		//double invActProb = 1./p.actionProb(s, a); //not needed in log gradient for labeled IRL...?
		FunctionGradient gradient = BoltzmannPolicyGradient.computeBoltzmannPolicyGradient(s, a, (DifferentiableQFunction)this.request.getPlanner(), this.request.getBoltzmannBeta());
		if(feedback == -1){
			for(FunctionGradient.PartialDerivative pd : gradient.getNonZeroPartialDerivatives()){
				gradient.put(pd.parameterId, -1.*pd.value);
			}
		}

		return gradient;
	}

	protected double probFeedback(double feedback, State s, Action a, Policy p){
		double pact = p.actionProb(s, a);
		if(feedback == 1.){
			return pact;
		}
		return 1. - pact;
	}

}
