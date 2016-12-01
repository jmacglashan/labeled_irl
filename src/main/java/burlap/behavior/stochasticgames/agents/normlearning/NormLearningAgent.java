package burlap.behavior.stochasticgames.agents.normlearning;

import burlap.behavior.policy.GreedyQPolicy;
import burlap.behavior.policy.Policy;
import burlap.behavior.policy.support.ActionProb;
import burlap.behavior.singleagent.Episode;
import burlap.behavior.singleagent.learnfromdemo.CustomRewardModel;
import burlap.behavior.singleagent.learnfromdemo.mlirl.MLIRL;
import burlap.behavior.singleagent.learnfromdemo.mlirl.MLIRLRequest;
import burlap.behavior.singleagent.learnfromdemo.mlirl.differentiableplanners.DifferentiableSparseSampling;
import burlap.behavior.singleagent.learnfromdemo.mlirl.support.DifferentiableRF;
import burlap.behavior.singleagent.planning.stochastic.sparsesampling.SparseSampling;
import burlap.behavior.stochasticgames.GameEpisode;
import burlap.behavior.stochasticgames.auxiliary.jointmdp.CentralizedDomainGenerator;
import burlap.behavior.stochasticgames.auxiliary.jointmdp.DecentralizedPolicy;
import burlap.behavior.stochasticgames.auxiliary.jointmdp.TotalWelfare;
import burlap.behavior.valuefunction.QFunction;
import burlap.behavior.valuefunction.ValueFunction;
import burlap.mdp.core.TerminalFunction;
import burlap.mdp.core.action.Action;
import burlap.mdp.core.state.State;
import burlap.mdp.singleagent.SADomain;
import burlap.mdp.singleagent.model.SampleModel;
import burlap.mdp.stochasticgames.JointAction;
import burlap.mdp.stochasticgames.SGDomain;
import burlap.mdp.stochasticgames.agent.SGAgent;
import burlap.mdp.stochasticgames.agent.SGAgentType;
import burlap.mdp.stochasticgames.model.JointRewardFunction;
import burlap.mdp.stochasticgames.world.World;
import burlap.statehashing.simple.SimpleHashableStateFactory;

import java.util.ArrayList;
import java.util.List;

/**
 * @author James MacGlashan
 */
public class NormLearningAgent implements SGAgent{

    protected SGDomain sgDomain;
    protected SADomain cmdpDomain;

    List<Episode> cgames = new ArrayList<Episode>();
    GameEpisode currentGame;

    public DecentralizedPolicy dp;
    public Policy learnedJoint;
    DecentralizedPolicy teamPolicy;
    ValueFunction normLeafValues;

    boolean started = false;
    boolean gameStart = false;
    boolean learnFromBadGames = false;

    protected int jointPlannerH;
    protected int RHIRL_h = 1;
    protected int c;

    protected DifferentiableRF normRF;


    protected double irlBoltzmannBeta = 1.;
    protected double irlLearningRate = 0.1;
    protected double irlLogLikelihoodChangeThreshold = 0.01;
    protected int irlMaxGradientAscentSteps = 10;

    protected World world;
    protected String agentName;
    protected SGAgentType type;
    protected int agentNum;




    /**
     * This constructor will automatically plan in the joint task to seed the RHIRL leaf node values by using
     * a {@link burlap.behavior.singleagent.planning.stochastic.sparsesampling.SparseSampling} planner.
     * RHIRL will by default use a horizon of 1 (use setter to change).
     * @param sgDomain a domain
     * @param normRF the parameterized joint task family in which to learn
     * @param jointPlannerH the horizon for planning the joint task
     * @param c the transition sampling size for the joint task planner and RHIRL. Use -1 for full Bellman
     */
    public NormLearningAgent(SGDomain sgDomain, DifferentiableRF normRF,
                             int jointPlannerH, int c, boolean learnFromBadGames) {
        this.sgDomain = sgDomain;
        this.normRF = normRF;
        this.jointPlannerH = jointPlannerH;
        this.c = c;

        this.learnFromBadGames = learnFromBadGames;
    }


    /**
     * This constructor will use the leaf node values in the RHIRL that are provided rather than creating a planner for it.
     * RHIRL will by default use a horizon of 1 (use setter to change).
     * @param sgDomain a domain
     * @param normRF the parameterized joint task family in which to learn
     * @param c the transition sampling size for RHIRL. Use -1 for full Bellman
     * @param leafValues the (potentially differentiable) leaf node values used in RHIRL
     */
    public NormLearningAgent(SGDomain sgDomain, DifferentiableRF normRF, int c,
                             ValueFunction leafValues, boolean learnFromBadGames) {
        this.sgDomain = sgDomain;
        this.normRF = normRF;
        this.c = c;
        this.normLeafValues = leafValues;

        this.learnFromBadGames = learnFromBadGames;
    }

    public NormLearningAgent copy() {
        return new NormLearningAgent(this.sgDomain, this.normRF, this.c,
                this.normLeafValues, this.learnFromBadGames);
    }

    /**
     * Use this to set the horizon used in RHIRL.
     * @param RHIRL_h
     */
    public void setRHIRL_h(int RHIRL_h) {
        this.RHIRL_h = RHIRL_h;
    }

    public SADomain getCmdpDomain(){
        return this.cmdpDomain;
    }


    /**
     * Sets the IRL Parameters this agent will use
     * @param irlBoltzmannBeta the Boltzmann distribution beta parameter; large is more deterministic; smaller is more uniform
     * @param irlLearningRate the gradient ascent learning rate/step size
     * @param irlLogLikelihoodChangeThreshold the threshold in log likelihood change to terminate gradient ascent
     * @param irlMaxGradientAscentSteps the maximum number of gradient ascent steps to take when running IRL
     */
    public void setIRLParameters(double irlBoltzmannBeta, double irlLearningRate, double irlLogLikelihoodChangeThreshold, int irlMaxGradientAscentSteps){
        this.irlBoltzmannBeta = irlBoltzmannBeta;
        this.irlLearningRate = irlLearningRate;
        this.irlLogLikelihoodChangeThreshold = irlLogLikelihoodChangeThreshold;
        this.irlMaxGradientAscentSteps = irlMaxGradientAscentSteps;
    }

    @Override
    public String agentName() {
        return this.agentName;
    }

    @Override
    public SGAgentType agentType() {
        return this.type;
    }

    @Override
    public void gameStarting(World w, int agentNum) {
        this.world = w;
        this.agentNum = agentNum;

        if(!this.started) {
            this.generateEquivelentSADomain();
            this.teamPolicy = this.generateNoExperiencePolicy();
            this.dp = this.teamPolicy;
            this.started = true;
        }

        this.gameStart = true;
    }

    @Override
    public Action action(State s) {
        return this.dp.action(s);
    }

    @Override
    public void observeOutcome(State s, JointAction jointAction, double[] jointReward, State sprime, boolean isTerminal) {

        if(this.gameStart){
            this.currentGame = new GameEpisode(s);
            System.out.println(this.currentGame);
            this.gameStart = false;
        }

        this.currentGame.transition(jointAction, sprime, jointReward);

    }

    @Override
    public void gameTerminated() {
        Episode ea = CentralizedDomainGenerator.gameAnalysisToEpisodeAnalysis(this.cmdpDomain, this.currentGame);
        double reward = ea.discountedReturn(1.0);
        if (reward > 0.0 || learnFromBadGames) {
            this.cgames.add(ea);
        }
        if (this.cgames.size() > 0) {
            this.dp = this.generateExperiencedPolicy(this.cgames);
        }
    }

    protected void generateEquivelentSADomain() {
        CentralizedDomainGenerator cmdpgen = new CentralizedDomainGenerator(this.sgDomain, this.world.getRewardFunction(), this.world.getTF(), new ArrayList<SGAgentType>(this.world.getAgentDefinitions()));
        this.cmdpDomain = cmdpgen.generateDomain();
    }

    protected DecentralizedPolicy generateNoExperiencePolicy() {
        JointRewardFunction jr = this.world.getRewardFunction();
        TerminalFunction tf = this.world.getTF();

        TotalWelfare crf = new TotalWelfare(jr);

        if(this.normLeafValues == null) {

            final SparseSampling cmdpTWPlanner = new SparseSampling(this.cmdpDomain, 0.99, new SimpleHashableStateFactory(), this.jointPlannerH, this.c);
            cmdpTWPlanner.setModel(new CustomRewardModel(this.cmdpDomain.getModel(), normRF));
            cmdpTWPlanner.toggleDebugPrinting(false);

            this.normLeafValues = new QFunction() {
                @Override
                public double qValue(State s, Action a) {
                    return cmdpTWPlanner.qValue(s, a);
                }

                @Override
                public double value(State s) {
                    return cmdpTWPlanner.value(s);
                }
            };
        }

        SparseSampling learnedPlanner = new SparseSampling(this.cmdpDomain, 0.99, new SimpleHashableStateFactory(), RHIRL_h, this.c);
        learnedPlanner.setValueForLeafNodes(this.normLeafValues);
        this.learnedJoint = new GreedyQPolicy(learnedPlanner);
        return new DecentralizedPolicy(new GreedyQPolicy(learnedPlanner), this.agentNum);
    }

    protected DecentralizedPolicy generateExperiencedPolicy(List<Episode> cgames) {
        System.out.println(this.agentName() + ": begin learning");

        SampleModel customRewardModel = new CustomRewardModel(this.cmdpDomain.getModel(), this.normRF);

        DifferentiableSparseSampling dss = new DifferentiableSparseSampling(this.cmdpDomain, normRF, 0.99, new SimpleHashableStateFactory(), this.RHIRL_h, this.c, this.irlBoltzmannBeta);
        dss.setModel(customRewardModel);
        dss.toggleDebugPrinting(false);
        dss.setValueForLeafNodes(this.normLeafValues);

        //now run IRL
        MLIRLRequest request = new MLIRLRequest(this.cmdpDomain, dss, cgames, this.normRF);
        request.setBoltzmannBeta(this.irlBoltzmannBeta);
        MLIRL irl = new MLIRL(request, this.irlLearningRate, this.irlLogLikelihoodChangeThreshold, this.irlMaxGradientAscentSteps);
        irl.toggleDebugPrinting(false);
        irl.performIRL();

        //setup our new policy
        SparseSampling learnedPlanner = new SparseSampling(this.cmdpDomain, 0.99, new SimpleHashableStateFactory(), this.RHIRL_h, this.c);
        learnedPlanner.setModel(customRewardModel);
        learnedPlanner.setValueForLeafNodes(this.normLeafValues);
        System.out.println(this.agentName() + ": end learning");
        this.learnedJoint = new GreedyQPolicy(learnedPlanner);
        return new DecentralizedPolicy(learnedJoint, this.agentNum);
    }

    public Policy getJointPolicy(){
        return this.learnedJoint;
    }

    public List<Episode> getcGames(){
        return this.cgames;
    }

}
