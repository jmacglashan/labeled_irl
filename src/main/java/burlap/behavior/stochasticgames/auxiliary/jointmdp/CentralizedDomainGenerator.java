package burlap.behavior.stochasticgames.auxiliary.jointmdp;


import burlap.behavior.singleagent.Episode;
import burlap.behavior.stochasticgames.GameEpisode;
import burlap.mdp.auxiliary.DomainGenerator;
import burlap.mdp.core.StateTransitionProb;
import burlap.mdp.core.TerminalFunction;
import burlap.mdp.core.action.Action;
import burlap.mdp.core.action.ActionType;
import burlap.mdp.core.state.State;
import burlap.mdp.singleagent.SADomain;
import burlap.mdp.singleagent.environment.EnvironmentOutcome;
import burlap.mdp.singleagent.model.FullModel;
import burlap.mdp.singleagent.model.RewardFunction;
import burlap.mdp.singleagent.model.TransitionProb;
import burlap.mdp.stochasticgames.JointAction;
import burlap.mdp.stochasticgames.SGDomain;
import burlap.mdp.stochasticgames.agent.SGAgentType;
import burlap.mdp.stochasticgames.model.FullJointModel;
import burlap.mdp.stochasticgames.model.JointRewardFunction;

import java.util.ArrayList;
import java.util.List;


/**
 * @author James MacGlashan.
 */
public class CentralizedDomainGenerator implements DomainGenerator {

	public static final String WRAPPED_ACTION = "wrappedAction";

	SGDomain srcDomain;
    JointRewardFunction jrf;
    TerminalFunction tf;
	List<SGAgentType> agentTypes;


	public CentralizedDomainGenerator(SGDomain srcDomain, JointRewardFunction jrf, TerminalFunction tf, List<SGAgentType> agentTypes) {
		this.srcDomain = srcDomain;
        this.jrf = jrf;
        this.tf = tf;
		this.agentTypes = agentTypes;
	}

	public SGDomain getSrcDomain() {
		return srcDomain;
	}

	public void setSrcDomain(SGDomain srcDomain) {
		this.srcDomain = srcDomain;
	}

	public List<SGAgentType> getAgentTypes() {
		return agentTypes;
	}

	public void setAgentTypes(List<SGAgentType> agentTypes) {
		this.agentTypes = agentTypes;
	}

	@Override
	public SADomain generateDomain() {

		SADomain domainWrapper = new SADomain();

        domainWrapper.addActionType(new JointActionTypeWrapper(this.agentTypes));
		domainWrapper.setModel(new CentralizedModel((FullJointModel)srcDomain.getJointActionModel(), this.jrf, this.tf));

		return domainWrapper;

	}

	public static List<Episode> gameAnalysesToEpisodeAnalyses(SADomain cmdp, List<GameEpisode> games){
		List<Episode> episodes = new ArrayList<Episode>(games.size());
		for(GameEpisode ga : games){
			episodes.add(gameAnalysisToEpisodeAnalysis(cmdp, ga));
		}
		return episodes;
	}

	public static Episode gameAnalysisToEpisodeAnalysis(SADomain cmdp, GameEpisode ga){

		Episode ea = new Episode(ga.state(0));

		for(int t = 0; t < ga.maxTimeStep(); t++){

			State nstate = ga.state(t+1);
			double [] rs = ga.jointReward(t+1);
			double sumReward = 0.;
			for(double d : rs){
				sumReward += d;
			}

			ea.transition(ga.jointAction(t), nstate, sumReward);
		}

		return ea;
	}

	public static class JointActionTypeWrapper implements ActionType{

        List<SGAgentType> agentTypes;

        public JointActionTypeWrapper(List<SGAgentType> agentTypes) {
            this.agentTypes = agentTypes;
        }

        @Override
        public String typeName() {
            return WRAPPED_ACTION;
        }

        @Override
        public Action associatedAction(String strRep) {
            throw new RuntimeException("Unimplemented");
        }

        @Override
        public List<Action> allApplicableActions(State s) {

            List<JointAction> jointActions = JointAction.getAllJointActionsFromTypes(s, agentTypes);
            List<Action> actions = new ArrayList<>(jointActions.size());
            for(JointAction ja : jointActions){
                actions.add(ja);
            }
            return actions;
        }


    }

    public static class CentralizedModel implements FullModel{

        public FullJointModel jmodel;
        public RewardFunction rf;
        public TerminalFunction tf;


        public CentralizedModel(FullJointModel jmodel, JointRewardFunction jrf, TerminalFunction tf) {
            this.jmodel = jmodel;
            this.tf = tf;
            this.rf = new TotalWelfare(jrf);
        }

        @Override
        public List<TransitionProb> transitions(State s, Action a) {

            JointAction ja = (JointAction)a;
            List<StateTransitionProb> stransitions = jmodel.stateTransitions(s, ja);
            List<TransitionProb> fullTransitions = new ArrayList<>(stransitions.size());
            for(StateTransitionProb tp : stransitions){
                double r = this.rf.reward(s, a, tp.s);
                boolean term = this.tf.isTerminal(tp.s);
                EnvironmentOutcome eo = new EnvironmentOutcome(s, a, tp.s, r, term);
                TransitionProb ntp = new TransitionProb(tp.p, eo);
                fullTransitions.add(ntp);
            }


            return fullTransitions;
        }

        @Override
        public EnvironmentOutcome sample(State s, Action a) {

            JointAction ja = (JointAction)a;
            State ns = this.jmodel.sample(s, ja);
            double r = this.rf.reward(s, a, ns);
            boolean term = this.terminal(ns);

            return new EnvironmentOutcome(s, a, ns, r, term);

        }

        @Override
        public boolean terminal(State s) {
            return this.tf.isTerminal(s);
        }
    }



}
