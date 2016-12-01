package burlap.behavior.singleagent.learnfromdemo.labeled;

import burlap.behavior.functionapproximation.FunctionGradient;
import burlap.behavior.policy.BoltzmannQPolicy;
import burlap.behavior.policy.Policy;
import burlap.behavior.singleagent.Episode;
import burlap.behavior.singleagent.learnfromdemo.CustomRewardModel;
import burlap.behavior.singleagent.learnfromdemo.mlirl.support.BoltzmannPolicyGradient;
import burlap.behavior.singleagent.learnfromdemo.mlirl.support.DifferentiableQFunction;
import burlap.behavior.singleagent.learnfromdemo.mlirl.support.DifferentiableRF;
import burlap.behavior.valuefunction.QProvider;
import burlap.datastructures.HashedAggregator;
import burlap.debugtools.DPrint;
import burlap.mdp.core.action.Action;
import burlap.mdp.core.state.State;

import java.util.*;

/**
 * @author James MacGlashan
 */
public class EMLabeledIRL {

    /**
     * The MLRIL request defining the IRL problem.
     */
    protected LabeledIRLRequest request;

    /**
     * The gradient ascent learning rate
     */
    protected double learningRate;

    protected int maxGASteps;

    protected int maxEMSteps;

    protected double phi = 7.;

    protected int numSampleScalar = 50;



    protected int debugCode = 27373;

    public boolean importanceSampling = true;


    public EMLabeledIRL(LabeledIRLRequest request, double learningRate, int gaSteps, int emSteps){

        this.request = request;
        this.learningRate = learningRate;
        this.maxGASteps = gaSteps;
        this.maxEMSteps = emSteps;


        if(!request.isValid()){
            throw new RuntimeException("Provided MLIRLRequest object is not valid.");
        }

    }

    public LabeledIRLRequest getRequest(){
        return this.request;
    }

    public void setNumSampleScalar(int nsamples){
        this.numSampleScalar = nsamples;
    }

    public void updateModelForCurrentReward(){
        DifferentiableRF rf = this.request.getRf();
        this.request.getPlanner().resetSolver();
        this.request.getPlanner().setModel(new CustomRewardModel(request.getDomain().getModel(), rf));
        //WARNING: Call planFromState from some trajectory state if you are going to use VI instead of sparse sampling
    }


    public void learn(){

        DifferentiableRF rf = this.request.getRf();

        this.updateModelForCurrentReward();

        System.out.println("Starting log likelihood: " + this.logLikelihood());

        for(int t = 0; t < this.maxEMSteps; t++){
            DPrint.cl(debugCode, "EM step " + (t+1));
            emStep();
            DPrint.cl(this.debugCode, "RF: " + this.request.getRf().toString());
        }


    }

    public void emStep(){

        DifferentiableRF rf = this.request.getRf();

        //generate samples and weights (E step)
        List<List<WeightedEpisode>> weightedSamples = new ArrayList<>(request.getExpertEpisodes().size());
        for(int i = 0; i < request.getExpertEpisodes().size(); i++){
            Episode e = request.getExpertEpisodes().get(i);
            double l = request.getEpisodeLabels().get(i);
            List<WeightedEpisode> samples = this.weightedAssignments(e, l);

            /**
             * inserting test code nakul
             */
            String str = "";

            for(int j=0;j<e.actionSequence.size();j++){
                str+=" "+e.actionSequence.get(j).actionName();
            }
            System.out.println(str);
            System.out.println("episode label: " + request.getEpisodeLabels().get(i));

            double[] avgLabel = new double[samples.get(0).e.rewardSequence.size()];
            double[] avgWeight = new double[samples.size()];
            double[] avgAbsWeight = new double[samples.size()];



            for(int j=0;j<samples.size();j++){
                WeightedEpisode we = samples.get(j);
                avgWeight[j]+=we.w;
                avgAbsWeight[j]+=Math.abs(we.w);
                for(int k=0;k<we.e.rewardSequence.size();k++){
                    if(we.e.rewardSequence.get(k)>0)
                    avgLabel[k]+=1;
                }
            }

            str = "";
            String str1 = "";

            for(int j=0;j<samples.size();j++){
//                WeightedEpisode we = samples.get(j);
                str+=", " + avgWeight[j];
                str1+=", " + avgAbsWeight[j];

            }



            System.out.println(str);
            System.out.println(str1);

            str = "";

            for(int k=0;k<avgLabel.length;k++){
                    str+= "," + avgLabel[k];
            }
            System.out.println(str);


            // end of nakul code
            weightedSamples.add(samples);
        }

        //now maximize with stochastic gradient ascent. Iterate n times over the batch
        for(int t = 0; t < maxGASteps; t++){

            DPrint.cl(debugCode, "    GA Batch " + (t+1));

            //for each demonstration's set of samples...
            int k = 1;
            for(List<WeightedEpisode> samples : weightedSamples){

                DPrint.cl(debugCode, "        Sample step " + k);

                //sum up the gradient of each sampled trajectory
                Policy p = this.curPolicy();
                HashedAggregator<Integer> sumGrad = new HashedAggregator<>();
                for(WeightedEpisode we : samples){
                    FunctionGradient grad_t = this.grad_trajectory(we.e, p);
                    scalarMult(grad_t, we.w);
                    sumInto(grad_t, sumGrad);
                }

                FunctionGradient grad_samples = toGradient(sumGrad);

                if(this.importanceSampling) {
                    //normalize the set of samples gradient
                    scalarMult(grad_samples, 1. / samples.size());
                }

                //move parameters in that direction
                for(FunctionGradient.PartialDerivative pd : grad_samples.getNonZeroPartialDerivatives()){
                    double curVal = rf.getParameter(pd.parameterId);
                    double nexVal = curVal + this.learningRate*pd.value;
                    rf.setParameter(pd.parameterId, nexVal);
                }

                //update planner for new reward parameters
                //reset valueFunction
                this.updateModelForCurrentReward();

                k++;

            }

            DPrint.cl(debugCode, "    log likelihood: " + this.logLikelihood());

        }

    }

    public double logLikelihood(){

        Policy p = this.curPolicy();
        double sum = 0.;
        for(int i = 0; i < request.getExpertEpisodes().size(); i++){
            Episode e = request.getExpertEpisodes().get(i);
            double label = request.getEpisodeLabels().get(i);

            DPAlgInfo info = computeDPInfo(e, p);
            LabelProbDP dp = new LabelProbDP(info.knownNet, info.transitionProbs, this.phi, e.numActions());
            double labelProb = dp.marginal(label);
            double logLabel = Math.log(labelProb);
            sum += logLabel;

            for(int t = 0; t < e.maxTimeStep(); t++){
                if(e.reward(t+1) != 0.0) {
                    double pFeedback = this.probFeedback(e.state(t), e.action(t), e.reward(t + 1), p);
                    double logP = Math.log(pFeedback);
                    sum += logP;
                }
            }
        }

        return sum;

    }

    protected List<WeightedEpisode> weightedAssignments(Episode e, double l){
        if(this.importanceSampling){
            return generateImportanceSamples(e, l);
        }
        else{
            return generateMarginalizedEpisodes(e, l);
        }
    }

    protected List<WeightedEpisode> generateImportanceSamples(Episode e, double l){

        int numUnknown = this.numUnknown(e);
        if(numUnknown == 0){
            return Arrays.asList(new WeightedEpisode(e, 1.));
        }

        Policy p = this.curPolicy();
        int numSamples = numSamples(numUnknown);
        List<Episode> samples = LabelAssignments.generateSamples(e, p, numSamples);
        List<WeightedEpisode> wsamples = new ArrayList<>(samples.size());
        for(Episode s : samples){
            double w = sampleWeight(l, s, e, p);
            wsamples.add(new WeightedEpisode(s, w));
        }

        return wsamples;

    }

    protected List<WeightedEpisode> generateMarginalizedEpisodes(Episode e, double l){

        int numUnknown = this.numUnknown(e);
        if(numUnknown == 0){
            return Arrays.asList(new WeightedEpisode(e, 1.));
        }

        List<Episode> marginalEpisodes = LabelAssignments.partialEnumerateEpisodeReward(e);
        Policy p = this.curPolicy();
        List<WeightedEpisode> wsamples = new ArrayList<>(marginalEpisodes.size());
        for(Episode s : marginalEpisodes){
            double w = fullEMWeight(l, s, e, p);
            wsamples.add(new WeightedEpisode(s, w));
        }

        return wsamples;
    }

    /**
     * Given the number of unknown feedbacks, how many importance samples should I use?
     */
    protected int numSamples(int numUnknown){
        if(numUnknown == 0) return numSampleScalar;
        return (int) (numSampleScalar * (Math.log(numUnknown+1) / Math.log(2)) );
    }

    protected Policy curPolicy(){
        Policy policy = new BoltzmannQPolicy((QProvider)this.request.getPlanner(), 1./request.getBoltzmannBeta());
        return policy;
    }

    protected Policy importanceSamplingPolicy(){
        Policy policy = new BoltzmannQPolicy((QProvider)this.request.getPlanner(), 1./request.getBoltzmannBeta()*10);
        return policy;
    }

    protected double sampleWeight(double l, Episode sample, Episode srcEpisode, Policy p){

        //compute weight numerator. For a given trajectory sample from importance distribution,
        //Pr(l | x_k, x_u)
        double netSample = sample.discountedReturn(1.);
        double pSample = probL(l, netSample, sample.numActions());


        double ratio = 1.;

        for(int t = 0; t < sample.maxTimeStep(); t++){
            if(sample.reward(t+1) != 0.0) {
                double pFeedback = this.probFeedback(sample.state(t), sample.action(t), sample.reward(t + 1), p);

                double impFeedback = this.probFeedback(sample.state(t), sample.action(t), sample.reward(t + 1), this.importanceSamplingPolicy());

//                double logP = Math.log(pFeedback);
                ratio*=pFeedback/impFeedback;
            }
        }


        //get DP parameter info
        DPAlgInfo info = computeDPInfo(srcEpisode, p);

        //do DP part: denominator of the weight: Pr(l | x_k)
        LabelProbDP dp = new LabelProbDP(info.knownNet, info.transitionProbs, this.phi, srcEpisode.numActions());
        double marginal = dp.marginal(l); //equation 11


        //now we can compute the weight: Pr(l | x_k, x_u) / Pr(l | x_k)

        double weight = pSample / marginal * ratio;

        return weight;

    }

    protected double fullEMWeight(double l, Episode sample, Episode srcEpisode, Policy p){

        //compute \prod_i \Pr(x_{u,i} | s_i, a_a, \theta)
        double prod = 1.;
        for(int t = 1; t <= srcEpisode.maxTimeStep(); t++){
            if(srcEpisode.reward(t) == 0.){
                double sampleFeedback = sample.reward(t);
                double pf = this.probFeedback(sample.state(t-1), sample.action(t-1), sampleFeedback, p);
                prod *= pf;
            }
        }

        double netSample = sample.discountedReturn(1.);
        double probSampleLabel = this.probL(l, netSample, sample.numActions());
        double numerator = probSampleLabel * prod;

        //now compute Pr(l | x_k, s, a, \theta)
        //get DP parameter info
        DPAlgInfo info = computeDPInfo(srcEpisode, p);

        //do DP part: denominator of the weight: Pr(l | x_k)
        LabelProbDP dp = new LabelProbDP(info.knownNet, info.transitionProbs, this.phi, srcEpisode.numActions());
        double marginal = dp.marginal(l);

        //final value is ( Pr(l | x_k, x_u) \prod_i \Pr(x_{u,i} | s_i, a_a, \theta) )  / Pr(l | x_k)
        double weight = numerator / marginal;

        return weight;
    }

    protected DPAlgInfo computeDPInfo(Episode e, Policy p){

        //build the DP +/- transition probabilities
        //and also sum up the value of the observed feedbacks: sum(X_k)
        double knownNet = 0.;
        List<Double> transitionProbsList = new ArrayList<>(e.numActions());
        for(int t = 1; t < e.numTimeSteps(); t++){
            double x = e.reward(t);
            if(x == 0.){
                double px = probFeedback(e.state(t-1), e.action(t-1), 1., p); //transition matrix always defined for probability of positive feedback
                transitionProbsList.add(px);
            }
            else{
                knownNet += x;
            }
        }
        //convert transition probs from a list to an array
        double [] transitionProbs = new double[transitionProbsList.size()];
        for(int i = 0; i < transitionProbsList.size(); i++){
            transitionProbs[i] = transitionProbsList.get(i);
        }

        DPAlgInfo info = new DPAlgInfo(knownNet, transitionProbs);
        return info;

    }

    protected double probL(double l, double net, int numFeedbacks){
        double sig = sigmoid(net, numFeedbacks);
        if(l == 1.){
            return sig;
        }
        return 1. - sig;
    }

    protected double sigmoid(double net, int numFeedbacks){
        // this is just weird this is just net/numFeedbacks to get a value between -1 and +1 as input to the sigmoid :)
        double norm = ((net + numFeedbacks) / (double)(2 * numFeedbacks));
        double scaled = norm - 0.5;
        double denom = 1. + Math.exp(-phi * scaled);
        double val = 1. / denom;
        return val;
    }

    protected double probFeedback(State s, Action a, double x, Policy p){
        if(x == 1){
            return p.actionProb(s, a);
        }
        return 1. - p.actionProb(s, a);
    }


    protected int numUnknown(Episode e){
        double num = 0;
        for(double r : e.rewardSequence){
            if(r == 0.){
                num += 1;
            }
        }
        return (int)num;
    }


    protected FunctionGradient grad_trajectory(Episode e, Policy p){

        HashedAggregator<Integer> sumGrad = new HashedAggregator<>();
        for(int t = 0; t < e.maxTimeStep(); t++){
            FunctionGradient grad_feedback = this.grad_feedback(e.state(t), e.action(t), e.reward(t+1), p);
            sumInto(grad_feedback, sumGrad);
        }

        FunctionGradient fg = toGradient(sumGrad);

        return fg;

    }

    /**
     * Gradient of a feedback. Equals gradient of P_xij in doc.
     * @param x the feedback value
     * @param s state
     * @param a action
     * @param p the policy on which to scale the the gradient
     * @return the gradient
     */
    protected FunctionGradient grad_feedback(State s, Action a, double x, Policy p){
        //double invActProb = 1./p.actionProb(s, a); //not needed in log gradient for labeled IRL...?
        FunctionGradient gradient = BoltzmannPolicyGradient.computeBoltzmannPolicyGradient(s, a, (DifferentiableQFunction)this.request.getPlanner(), this.request.getBoltzmannBeta());
        double scalar;
        if(x == 1.){
            scalar = 1. / p.actionProb(s, a);
        }
        else{
            scalar = -1. / (1. - p.actionProb(s, a));
        }
        scalarMult(gradient, scalar);

        return gradient;
    }







    protected static FunctionGradient toGradient(HashedAggregator<Integer> summedParams){
        FunctionGradient fg = new FunctionGradient.SparseGradient(summedParams.size());
        for(Map.Entry<Integer, Double> e : summedParams.entrySet()){
            fg.put(e.getKey(), e.getValue());
        }
        return fg;
    }

    public static void scalarMult(FunctionGradient fg, double scalar){
        for(FunctionGradient.PartialDerivative pd : fg.getNonZeroPartialDerivatives()){
            double scaled = pd.value * scalar;
            fg.put(pd.parameterId, scaled);
        }
    }

    public static FunctionGradient scalarMultCopy(FunctionGradient fg, double scalar){
        FunctionGradient cfg = new FunctionGradient.SparseGradient(fg.numNonZeroPDs());
        for(FunctionGradient.PartialDerivative pd : fg.getNonZeroPartialDerivatives()){
            double scaled = pd.value * scalar;
            cfg.put(pd.parameterId, scaled);
        }
        return cfg;
    }



    protected static void sumInto(FunctionGradient fg, HashedAggregator<Integer> sum){
        for(FunctionGradient.PartialDerivative pd : fg.getNonZeroPartialDerivatives()){
            sum.add(pd.parameterId, pd.value);
        }
    }

    protected static FunctionGradient diffGrad(FunctionGradient a, FunctionGradient b){
        Set<Integer> pIds = pdIdSet(a, b);

        //now compute
        FunctionGradient fg = new FunctionGradient.SparseGradient(pIds.size());
        for(int pid : pIds){
            double v = a.getPartialDerivative(pid) - b.getPartialDerivative(pid);
            fg.put(pid, v);
        }

        return fg;

    }

    protected static FunctionGradient addGrad(FunctionGradient a, FunctionGradient b){

        Set<Integer> pIds = pdIdSet(a, b);

        //now compute
        FunctionGradient fg = new FunctionGradient.SparseGradient(pIds.size());
        for(int pid : pIds){
            double v = a.getPartialDerivative(pid) + b.getPartialDerivative(pid);
            fg.put(pid, v);
        }

        return fg;

    }

    protected static Set<Integer> pdIdSet(FunctionGradient a, FunctionGradient b){
        Set<FunctionGradient.PartialDerivative> aSet = a.getNonZeroPartialDerivatives();
        Set<FunctionGradient.PartialDerivative> bSet = b.getNonZeroPartialDerivatives();
        Set<Integer> pIds = new HashSet<>(aSet.size()+bSet.size());
        for(FunctionGradient.PartialDerivative pd : aSet){
            pIds.add(pd.parameterId);
        }
        for(FunctionGradient.PartialDerivative pd : bSet){
            pIds.add(pd.parameterId);
        }

        return pIds;
    }



    protected static class WeightedEpisode{
        public Episode e;
        public double w;

        public WeightedEpisode(Episode e, double w) {
            this.e = e;
            this.w = w;
        }
    }

    protected static class DPAlgInfo{
        public double knownNet;
        public double [] transitionProbs;

        public DPAlgInfo(double knownNet, double[] transitionProbs) {
            this.knownNet = knownNet;
            this.transitionProbs = transitionProbs;
        }
    }

}
