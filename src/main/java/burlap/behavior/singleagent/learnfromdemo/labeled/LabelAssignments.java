package burlap.behavior.singleagent.learnfromdemo.labeled;

import burlap.behavior.policy.Policy;
import burlap.behavior.policy.RandomPolicy;
import burlap.behavior.singleagent.Episode;
import burlap.debugtools.RandomFactory;
import burlap.mdp.core.action.Action;
import burlap.mdp.core.action.ActionType;
import burlap.mdp.core.action.SimpleAction;
import burlap.mdp.core.action.UniversalActionType;
import burlap.mdp.core.state.NullState;
import burlap.mdp.core.state.State;

import java.util.*;

/**
 * @author James MacGlashan.
 */
public class LabelAssignments {


    /**
     * Generates episodes for all possible reward assignments for *every* step of the
     * episode
     * @param e the original episode
     * @return a list of episodes
     */
	public static List<Episode> fullEnumerateEpisodeReward(Episode e){

		int n = e.numActions();
		List<double[]> assignments = generateAssignments(n);
		List<Episode> episodes = new ArrayList<>(assignments.size());
		for(double [] assignment : assignments){
			Episode ne = e.copy();
			for(int t = 1; t < ne.numTimeSteps(); t++){
				ne.rewardSequence.set(t-1, assignment[t-1]);
			}
			episodes.add(ne);
		}

		return episodes;

	}


    /**
     * Generates episodes for all possible reward assignments (in {-1, 1}) for steps in the original
     * episode that have a reward value of 0.
     * @param e the source episode from new episode reward assignments are generated
     * @return a list of episodes
     */
	public static List<Episode> partialEnumerateEpisodeReward(Episode e){
	    List<Integer> freeRewards = freeRewards(e);
        int n = freeRewards.size();
        List<double[]> assignments = generateAssignments(n);
        List<Episode> episodes = new ArrayList<>(assignments.size());
        for(double [] assignment : assignments){
            Episode ne = e.copy();
            for(int i = 0; i < assignment.length; i++){
                int ind = freeRewards.get(i);
                ne.rewardSequence.set(ind, assignment[i]);
            }
            episodes.add(ne);
        }

        return episodes;
    }

    public static List<Integer> freeRewards(Episode e){
        List<Integer> freeRewards = new ArrayList<>(e.rewardSequence.size());
        for(int i = 0; i < e.rewardSequence.size(); i++){
            if(e.rewardSequence.get(i) == 0.){
                freeRewards.add(i);
            }
        }
        return freeRewards;
    }

	public static List<double[]> generateAssignments(int n){

		List<double[]> results = new ArrayList<>();
		assignmentHelper(n, 0, new double[n], results);
		return results;

	}

	public static void assignmentHelper(int n, int i, double [] cur, List<double[]> results){

		if(i == n){
			results.add(cur.clone());
			return ;
		}

		cur[i] = 1.;
		assignmentHelper(n, i+1, cur, results);

		cur[i] = -1.;
		assignmentHelper(n, i+1, cur, results);

	}

	public static int numUnlabeled(Episode e){
	    int num = 0;
        for(double r : e.rewardSequence){
            if(r != 0){
                num++;
            }
        }
        return num;
    }



    public static List<Episode> generateSamples(Episode srcEpisode, Policy p, int num){
        List<Episode> episodes = new ArrayList<>(num);
        for(int i = 0; i < num; i++){
            Episode e = srcEpisode.copy();
            giveRandomAssignments(e, p);
            episodes.add(e);
        }
        return episodes;
    }

    public static void giveRandomAssignments(Episode e, Policy p){
        Random rand = RandomFactory.getMapped(0);

        for(int i = 0; i < e.rewardSequence.size(); i++){
            if(e.rewardSequence.get(i) != 0){
                continue;
            }
            double roll = rand.nextDouble();
            double r = roll < p.actionProb(e.state(i), e.action(i)) ? 1. : -1.;
            e.rewardSequence.set(i, r);
        }
    }

	public static List<Episode> generateSamples(Episode srcEpisode, int num){
	    List<Episode> episodes = new ArrayList<>(num);
        for(int i = 0; i < num; i++){
            Episode e = srcEpisode.copy();
            giveRandomAssignments(e);
            episodes.add(e);
        }
        return episodes;
    }

	public static void giveRandomAssignments(Episode e){

	    Random rand = RandomFactory.getMapped(0);

        for(int i = 0; i < e.rewardSequence.size(); i++){
            if(e.rewardSequence.get(i) != 0){
                continue;
            }
            double roll = rand.nextInt(2) == 1 ? 1. : -1.;
            e.rewardSequence.set(i, roll);
        }

	}


    public static void main(String[] args) {

        Episode te = new Episode(NullState.instance);
        te.transition(new SimpleAction("act1"), NullState.instance, 0.);
        te.transition(new SimpleAction("act1"), NullState.instance, 1.);
        te.transition(new SimpleAction("act1"), NullState.instance, -1.);
        te.transition(new SimpleAction("act1"), NullState.instance, 0.);

        List<Episode> enumEps = partialEnumerateEpisodeReward(te);
        System.out.println(enumEps.size());
        System.out.println("---");
        for(Episode e : enumEps){
            for(int i = 0; i < e.rewardSequence.size(); i++){
                System.out.print(e.rewardSequence.get(i) + ", ");
            }
            System.out.println();
        }

        System.out.println("====");
        //Policy p = new RandomPolicy(Arrays.<ActionType>asList(new UniversalActionType("act1"), new UniversalActionType("act2")));
        Policy p = new Policy() {
            @Override
            public Action action(State s) {
                return new SimpleAction("act1");
            }

            @Override
            public double actionProb(State s, Action a) {
                double prob = .2;
                double val = a.actionName().equals("act1") ? prob : 1. - prob;
                return val;
            }

            @Override
            public boolean definedFor(State s) {
                return true;
            }
        };

        List<Episode> sampled = generateSamples(te, p, 10);
        for(Episode e : sampled){
            for(int i = 0; i < e.rewardSequence.size(); i++){
                System.out.print(e.rewardSequence.get(i) + ", ");
            }
            System.out.println();
        }

    }

}
