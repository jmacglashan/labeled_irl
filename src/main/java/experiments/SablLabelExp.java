package experiments;

import burlap.behavior.functionapproximation.dense.DenseStateFeatures;
import burlap.behavior.policy.GreedyQPolicy;
import burlap.behavior.singleagent.Episode;
import burlap.behavior.singleagent.auxiliary.EpisodeSequenceVisualizer;
import burlap.behavior.singleagent.auxiliary.StateReachability;
import burlap.behavior.singleagent.auxiliary.valuefunctionvis.ValueFunctionVisualizerGUI;
import burlap.behavior.singleagent.learnfromdemo.RewardValueProjection;
import burlap.behavior.singleagent.learnfromdemo.labeled.SablLabel;
import burlap.behavior.singleagent.learnfromdemo.mlirl.MLIRL;
import burlap.behavior.singleagent.learnfromdemo.mlirl.MLIRLRequest;
import burlap.behavior.singleagent.learnfromdemo.mlirl.commonrfs.LinearStateDifferentiableRF;
import burlap.behavior.singleagent.learnfromdemo.mlirl.differentiableplanners.DifferentiableSparseSampling;
import burlap.behavior.singleagent.learnfromdemo.mlirl.differentiableplanners.dpoperator.SubDifferentiableMaxOperator;
import burlap.behavior.valuefunction.QProvider;
import burlap.debugtools.RandomFactory;
import burlap.domain.singleagent.gridworld.GridWorldDomain;
import burlap.domain.singleagent.gridworld.GridWorldVisualizer;
import burlap.domain.singleagent.gridworld.state.GridAgent;
import burlap.domain.singleagent.gridworld.state.GridLocation;
import burlap.domain.singleagent.gridworld.state.GridWorldState;
import burlap.mdp.auxiliary.StateGenerator;
import burlap.mdp.core.oo.OODomain;
import burlap.mdp.core.oo.propositional.GroundedProp;
import burlap.mdp.core.oo.propositional.PropositionalFunction;
import burlap.mdp.core.oo.state.OOState;
import burlap.mdp.core.state.State;
import burlap.mdp.singleagent.environment.SimulatedEnvironment;
import burlap.mdp.singleagent.oo.OOSADomain;
import burlap.shell.visual.VisualExplorer;
import burlap.statehashing.simple.SimpleHashableStateFactory;
import burlap.visualizer.Visualizer;

import java.util.List;

/**
 * @author James MacGlashan.
 */
public class SablLabelExp {

	GridWorldDomain gwd;
	OOSADomain domain;
	StateGenerator sg;
	Visualizer v;

	public SablLabelExp(){

		this.gwd = new GridWorldDomain(5 ,5);
		this.gwd.setNumberOfLocationTypes(5);
		gwd.makeEmptyMap();
		this.domain = gwd.generateDomain();
		State bs = this.basicState();
		this.sg = new LabeledIRLExp.LeftSideGen(5, bs);
		this.v = GridWorldVisualizer.getVisualizer(this.gwd.getMap());

	}

	/**
	 * Creates a visual explorer that you can use to to record trajectories. Use the "`" key to reset to a random initial state
	 * Use the wasd keys to move north south, east, and west, respectively. To enable recording,
	 * first open up the shell and type: "rec -b" (you only need to type this one). Then you can move in the explorer as normal.
	 * Each demonstration begins after an environment reset.
	 * After each demonstration that you want to keep, go back to the shell and type "rec -r"
	 * If you reset the environment before you type that,
	 * the episode will be discarded. To temporarily view the episodes you've created, in the shell type "episode -v". To actually record your
	 * episodes to file, type "rec -w path/to/save/directory base_file_name" For example "rec -w irl_demos demo"
	 * A recommendation for examples is to record two demonstrations that both go to the pink cell while avoiding blue ones
	 * and do so from two different start locations on the left (if you keep resetting the environment, it will change where the agent starts).
	 */
	public void launchExplorer(){
		SimulatedEnvironment env = new SimulatedEnvironment(this.domain, this.sg);
		VisualExplorer exp = new VisualExplorer(this.domain, env, this.v, 800, 800);
		exp.addKeyAction("w", GridWorldDomain.ACTION_NORTH, "");
		exp.addKeyAction("s", GridWorldDomain.ACTION_SOUTH, "");
		exp.addKeyAction("d", GridWorldDomain.ACTION_EAST, "");
		exp.addKeyAction("a", GridWorldDomain.ACTION_WEST, "");

		//exp.enableEpisodeRecording("r", "f", "irlDemo");

		exp.initGUI();
	}


	/**
	 * Launch a episode sequence visualizer to display the saved trajectories in the folder "irlDemo"
	 */
	public void launchSavedEpisodeSequenceVis(String pathToEpisodes){

		new EpisodeSequenceVisualizer(this.v, this.domain, pathToEpisodes);

	}

	/**
	 * Runs MLIRL on the trajectories stored in the "irlDemo" directory and then visualizes the learned reward function.
	 */
	public void runIRL(String pathToEpisodes, int mode){

		//create reward function features to use
		LabeledIRLExp.LocationFeatures features = new LabeledIRLExp.LocationFeatures(this.domain, 5);

		//create a reward function that is linear with respect to those features and has small random
		//parameter values to start
		LinearStateDifferentiableRF rf = new LinearStateDifferentiableRF(features, 5);
		for(int i = 0; i < rf.numParameters(); i++){
			rf.setParameter(i, RandomFactory.getMapped(0).nextDouble()*0.2 - 0.1);
		}

		//load our saved demonstrations from disk
		List<Episode> episodes = Episode.readEpisodes(pathToEpisodes);
		//assign rewards to be good
		//this.assignRewards(episodes, -1.);
		this.assignRewards(episodes.get(0), 1.);
        this.assignRewards(episodes.get(1), -1.);

		//use either DifferentiableVI or DifferentiableSparseSampling for planning. The latter enables receding horizon IRL,
		//but you will probably want to use a fairly large horizon for this kind of reward function.
		double beta = 10;
		//DifferentiableVI dplanner = new DifferentiableVI(this.domain, rf, 0.99, beta, new SimpleHashableStateFactory(), 0.01, 100);
		DifferentiableSparseSampling dplanner = new DifferentiableSparseSampling(this.domain, rf, 0.99, new SimpleHashableStateFactory(), 2, -1, beta);
		dplanner.setOperator(new SubDifferentiableMaxOperator());

		dplanner.toggleDebugPrinting(false);

		MLIRLRequest request = null;
		//define the IRL problem
		request = new MLIRLRequest(domain, dplanner, episodes, rf);
		request.setBoltzmannBeta(beta);

		if(mode == 0){
			System.out.println("IRL!");
			//run MLIRL on it
			MLIRL irl = new MLIRL(request, 0.1, 0.1, 10);
			irl.performIRL();
		}
		else if(mode == 1){
			System.out.println("Sabl Label!");
			//run sabl label
			SablLabel irl = new SablLabel(request, 0.1, 0.001, 10);
			irl.performIRL();

		}



		//get all states in the domain so we can visualize the learned reward function for them
		List<State> allStates = StateReachability.getReachableStates(basicState(), this.domain, new SimpleHashableStateFactory());

		//get a standard grid world value function visualizer, but give it StateRewardFunctionValue which returns the
		//reward value received upon reaching each state which will thereby let us render the reward function that is
		//learned rather than the value function for it.
		ValueFunctionVisualizerGUI gui = GridWorldDomain.getGridWorldValueFunctionVisualization(
				allStates,
				5,
				5,
				new RewardValueProjection(rf),
				new GreedyQPolicy((QProvider) request.getPlanner())
		);

		gui.initGUI();


	}

	public void assignRewards(List<Episode> episodes, double r){
		for(Episode e : episodes){
			assignRewards(e, r);
		}
	}

	public void assignRewards(Episode e, double r){
		for(int i = 0; i < e.rewardSequence.size(); i++){
			e.rewardSequence.set(i, r);
		}
	}


	/**
	 * Creates a grid world state with the agent in (0,0) and various different grid cell types scattered about.
	 * @return a grid world state with the agent in (0,0) and various different grid cell types scattered about.
	 */
	protected State basicState(){

		GridWorldState s = new GridWorldState(
				new GridAgent(0, 0),
				new GridLocation(0, 0, 1, "loc0"),
				new GridLocation(0, 4, 2, "loc1"),
				new GridLocation(4, 4, 3, "loc2"),
				new GridLocation(4, 0, 4, "loc3"),

				new GridLocation(1, 0, 0, "loc4"),
				new GridLocation(1, 2, 0, "loc5"),
				new GridLocation(1, 4, 0, "loc6"),
				new GridLocation(3, 1, 0, "loc7"),
				new GridLocation(3, 3, 0, "loc8")
		);

		return s;
	}

	/**
	 * State generator that produces initial agent states somewhere on the left side of the grid.
	 */
	public static class LeftSideGen implements StateGenerator {


		protected int height;
		protected State sourceState;


		public LeftSideGen(int height, State sourceState){
			this.setParams(height, sourceState);
		}

		public void setParams(int height, State sourceState){
			this.height = height;
			this.sourceState = sourceState;
		}



		public State generateState() {

			GridWorldState s = (GridWorldState)this.sourceState.copy();

			int h = RandomFactory.getDefault().nextInt(this.height);
			s.touchAgent().y = h;

			return s;
		}
	}

	/**
	 * A state feature vector generator that create a binary feature vector where each element
	 * indicates whether the agent is in a cell of of a different type. All zeros indicates
	 * that the agent is in an empty cell.
	 */
	public static class LocationFeatures implements DenseStateFeatures {

		protected int numLocations;
		PropositionalFunction inLocationPF;


		public LocationFeatures(OODomain domain, int numLocations){
			this.numLocations = numLocations;
			this.inLocationPF = domain.propFunction(GridWorldDomain.PF_AT_LOCATION);
		}

		public LocationFeatures(int numLocations, PropositionalFunction inLocationPF) {
			this.numLocations = numLocations;
			this.inLocationPF = inLocationPF;
		}

		@Override
		public double[] features(State s) {

			double [] fv = new double[this.numLocations];

			int aL = this.getActiveLocationVal((OOState)s);
			if(aL != -1){
				fv[aL] = 1.;
			}

			return fv;
		}


		protected int getActiveLocationVal(OOState s){

			List<GroundedProp> gps = this.inLocationPF.allGroundings(s);
			for(GroundedProp gp : gps){
				if(gp.isTrue(s)){
					GridLocation l = (GridLocation)s.object(gp.params[1]);
					return l.type;
				}
			}

			return -1;
		}

		@Override
		public DenseStateFeatures copy() {
			return new LabeledIRLExp.LocationFeatures(numLocations, inLocationPF);
		}
	}

	public static void main(String[] args) {

		RandomFactory.seedMapped(0, 123);

		SablLabelExp ex = new SablLabelExp();

		//only have one of the below uncommented

		//ex.launchExplorer(); //choose this to record demonstrations
		//ex.launchSavedEpisodeSequenceVis("irl_demos2"); //choose this review the demonstrations that you've recorded
		ex.runIRL("irl_demos2", 1); //choose this to run MLIRL on the demonstrations and visualize the learned reward function and policy


	}


}
