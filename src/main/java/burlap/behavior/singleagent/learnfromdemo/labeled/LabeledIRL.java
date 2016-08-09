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
 *
 * Labeled IRL in which we use methods that look like our math
 *
 * @author James MacGlashan
 */
public class LabeledIRL {

    /**
     * The MLRIL request defining the IRL problem.
     */
    protected LabeledIRLRequest request;

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
     * The debug code used for printing information to the terminal.
     */
    protected int debugCode = 625420;

    /**
     * Initializes.
     * @param request the problem request definition
     * @param learningRate the gradient ascent learning rate
     * @param maxLikelihoodChange the likelihood change threshold that must be reached to terminate gradient ascent
     * @param maxSteps the maximum number of gradient ascent steps allowed before termination is forced. Set to -1 to rely only on likelihood threshold.
     */
    public LabeledIRL(LabeledIRLRequest request, double learningRate, double maxLikelihoodChange, int maxSteps){

        this.request = request;
        this.learningRate = learningRate;
        this.maxLikelihoodChange = maxLikelihoodChange;
        this.maxSteps = maxSteps;

        if(!request.isValid()){
            throw new RuntimeException("Provided MLIRLRequest object is not valid.");
        }

    }


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
            FunctionGradient gradient = this.grad_logLikelihood();
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


    protected FunctionGradient grad_logLikelihood(){

        HashedAggregator<Integer> sum = new HashedAggregator<>();
        for(int i = 0; i < this.request.getExpertEpisodes().size(); i++){
            FunctionGradient grad_logDi = this.grad_logDi(request.getExpertEpisodes().get(i), request.getEpisodeLabels().get(i));
            sumInto(grad_logDi, sum);
        }

        FunctionGradient finalGrad = toGradient(sum);

        return finalGrad;

    }

    protected FunctionGradient grad_logDi(Episode e, double l){

        //first get marginalized episodes
        List<Episode> X = LabelAssignments.fullEnumerateEpisodeReward(e);
        List<Double> Gxi = this.Gxi(X, l);
        int mi = arg_max(Gxi);
        List<Double> Fxi = this.Fxi(Gxi, mi);

        List<FunctionGradient> grad_Gxi = this.grad_Gxi(X);
        List<FunctionGradient> grad_Fxi = this.grad_Fxi(grad_Gxi, mi);

        double sumf = sumExponential(Fxi);

        HashedAggregator<Integer> sumGradAgg = new HashedAggregator<>();
        for(int i = 0; i < X.size(); i++){
            double expf = Math.exp(Fxi.get(i));
            FunctionGradient grad_fxi = grad_Fxi.get(i);
            FunctionGradient scaled = scalarMultCopy(grad_fxi, expf);
            sumInto(scaled, sumGradAgg);
        }

        FunctionGradient sumGrad = toGradient(sumGradAgg);
        scalarMult(sumGrad, 1. / sumf);

        FunctionGradient grad_gmii = grad_Gxi.get(mi);

        FunctionGradient total = addGrad(sumGrad, grad_gmii);


        return total;

    }

    protected List<FunctionGradient> grad_Fxi(List<FunctionGradient> grad_Gxi, int mi){

        List<FunctionGradient> res = new ArrayList<>(grad_Gxi.size());
        FunctionGradient gmii = grad_Gxi.get(mi);
        for(FunctionGradient grad_gxi : grad_Gxi){
            FunctionGradient diff = diffGrad(grad_gxi, gmii);
            res.add(diff);
        }

        return res;
    }


    protected List<FunctionGradient> grad_Gxi(List<Episode> X){
        List<FunctionGradient> fgs = new ArrayList<>(X.size());
        for(Episode e : X){
            FunctionGradient fg = grad_gxi(e);
            fgs.add(fg);
        }
        return fgs;
    }


    protected FunctionGradient grad_gxi(Episode e){


        Policy policy = new BoltzmannQPolicy((QProvider)this.request.getPlanner(), 1./request.getBoltzmannBeta());

        HashedAggregator<Integer> summed = new HashedAggregator<>();
        for(int t = 0; t < e.maxTimeStep(); t++){
            State s = e.state(t);
            Action a = e.action(t);
            double x = e.reward(t+1);
            FunctionGradient grad_pxij = this.grad_Pxij(s, a, x);
            double pxij = this.pxij(s, a, x, policy);
            scalarMult(grad_pxij, 1. / pxij);
            sumInto(grad_pxij, summed);
        }

        FunctionGradient finalGrad = toGradient(summed);

        return finalGrad;
    }



    /**
     * Gradient of a feedback. Equals gradient of P_xij in doc.
     * @param x the feedback value
     * @param s state
     * @param a action
     * @return the gradient
     */
    protected FunctionGradient grad_Pxij(State s, Action a, double x){
        //double invActProb = 1./p.actionProb(s, a); //not needed in log gradient for labeled IRL...?
        FunctionGradient gradient = BoltzmannPolicyGradient.computeBoltzmannPolicyGradient(s, a, (DifferentiableQFunction)this.request.getPlanner(), this.request.getBoltzmannBeta());
        if(x == -1){
            for(FunctionGradient.PartialDerivative pd : gradient.getNonZeroPartialDerivatives()){
                gradient.put(pd.parameterId, -1.*pd.value);
            }
        }

        return gradient;
    }











    /////////////////Non gradient loglikelihood functions/////////////////////

    protected double logLikelihood(){


        double sum = 0.;
        for(int i = 0; i < this.request.getExpertEpisodes().size(); i++){
            double logDi = this.logDi(request.getExpertEpisodes().get(i), request.getEpisodeLabels().get(i));
            sum += logDi;
        }

        return sum;

    }


    protected double logDi(Episode e, double l){

        //first get marginalized episodes
        List<Episode> X = LabelAssignments.fullEnumerateEpisodeReward(e);
        List<Double> Gxi = this.Gxi(X, l);
        int mi = arg_max(Gxi);
        List<Double> Fxi = this.Fxi(Gxi, mi);

        double gmi = Gxi.get(mi);

        double sumf = sumExponential(Fxi);
        double logsum = Math.log(sumf);

        double total = gmi + logsum;

        return total;

    }


    protected List<Double> Fxi(List<Double> Gxi, int mi){
        List<Double> res = new ArrayList<>(Gxi.size());
        double gmii = Gxi.get(mi);
        for(double gxi : Gxi){
            double diff = gxi - gmii;
            res.add(diff);
        }
        return res;
    }

    protected List<Double> Gxi(List<Episode> X, double l){
        List<Double> res = new ArrayList<>(X.size());
        for(Episode e : X){
            double gxi = this.gxi(e, l);
            res.add(gxi);
        }
        return res;
    }

    protected double gxi(Episode e, double l){
        double plxi = this.plix(e, l);
        double log_plxi = Math.log(plxi);

        Policy policy = new BoltzmannQPolicy((QProvider)this.request.getPlanner(), 1./request.getBoltzmannBeta());
        double sum = 0.;
        for(int t = 0; t < e.maxTimeStep(); t++){
            State s = e.state(t);
            Action a = e.action(t);
            double x = e.reward(t+1);
            double pxij = this.pxij(s, a, x, policy);
            double log_pxij = Math.log(pxij);
            sum += log_pxij;
        }

        double total = log_plxi + sum;
        return total;

    }

    protected double plix(Episode e, double label){
        double netx = this.netx(e);
        double sig = this.sigmoid(netx);
        if(label == 1.){
            return sig;
        }
        return 1. - sig;
    }

    protected double netx(Episode e){
        double sum = 0.;
        for(double r : e.rewardSequence){
            sum += r;
        }
        return phi * sum;
    }

    protected double pxij(State s, Action a, double x, Policy policy){
        if(x == 1.){
            return policy.actionProb(s, a);
        }
        return 1. - policy.actionProb(s, a);
    }

    protected double sigmoid(double x){
        return 1. / (1 + Math.exp(-x));
    }


    protected int arg_max(List<Double> vals){
        int ind = 0;
        double mx = vals.get(0);
        for(int i = 1; i < vals.size(); i++){
            if(vals.get(i) > mx){
                mx = vals.get(i);
                ind = i;
            }
        }
        return ind;
    }

    protected static double sumExponential(List<Double> values){

        double sum = 0.;
        for(double d : values){
            double exped = Math.exp(d);
            sum += exped;
        }

        return sum;

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



}
