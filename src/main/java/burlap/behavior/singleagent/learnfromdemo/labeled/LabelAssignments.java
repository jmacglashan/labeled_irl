package burlap.behavior.singleagent.learnfromdemo.labeled;

import burlap.behavior.singleagent.Episode;

import java.util.ArrayList;
import java.util.List;

/**
 * @author James MacGlashan.
 */
public class LabelAssignments {


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

}
