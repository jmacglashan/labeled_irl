package burlap.behavior.singleagent.learnfromdemo.labeled;

import burlap.behavior.singleagent.Episode;
import burlap.behavior.singleagent.learnfromdemo.mlirl.MLIRLRequest;
import burlap.behavior.singleagent.learnfromdemo.mlirl.support.DifferentiableRF;
import burlap.behavior.singleagent.planning.Planner;
import burlap.mdp.singleagent.SADomain;
import burlap.statehashing.HashableStateFactory;

import java.util.List;

/**
 * @author James MacGlashan.
 */
public class LabeledIRLRequest extends MLIRLRequest {

	protected List<Double> episodeLabels;

	public LabeledIRLRequest(SADomain domain, Planner planner, List<Episode> expertEpisodes, DifferentiableRF rf, List<Double> episodeLabels) {
		super(domain, planner, expertEpisodes, rf);
		this.episodeLabels = episodeLabels;
	}

	public LabeledIRLRequest(SADomain domain, List<Episode> expertEpisodes, DifferentiableRF rf, HashableStateFactory hashingFactory, List<Double> episodeLabels) {
		super(domain, expertEpisodes, rf, hashingFactory);
		this.episodeLabels = episodeLabels;
	}

	public List<Double> getEpisodeLabels() {
		return episodeLabels;
	}

	public void setEpisodeLabels(List<Double> episodeLabels) {
		this.episodeLabels = episodeLabels;
	}
}
