package burlap.behavior.singleagent.learnfromdemo.labeled;

import burlap.behavior.singleagent.Episode;

import java.util.List;

/**
 * @author James MacGlashan
 */
public class SampledLabeledIRL extends LabeledIRL {


    protected int sampleRate = 5;

    public SampledLabeledIRL(LabeledIRLRequest request, double learningRate, double maxLikelihoodChange, int maxSteps) {
        super(request, learningRate, maxLikelihoodChange, maxSteps);
    }




    protected List<Episode> generateSamples(Episode e){
        int numSamples = this.numSamplesFor(e);
        List<Episode> res = LabelAssignments.generateSamples(e, numSamples);
        return res;
    }


    protected int numSamplesFor(Episode e){
        int numUnlabeled = LabelAssignments.numUnlabeled(e);
        if(numUnlabeled == 0){
            return 1;
        }
        if(numUnlabeled == 1){
            return sampleRate;
        }

        return sampleRate * numUnlabeled * (int)Math.log(numUnlabeled);
    }

}
