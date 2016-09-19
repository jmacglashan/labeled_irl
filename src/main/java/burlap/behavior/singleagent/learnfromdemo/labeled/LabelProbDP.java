package burlap.behavior.singleagent.learnfromdemo.labeled;

import java.util.ArrayList;
import java.util.List;

/**
 * @author James MacGlashan
 */
public class LabelProbDP {

    protected double knownSum;
    protected int numUnknown;
    protected double [][] dpTable;
    protected int range;
    protected double [] transitionProbs;
    protected double phi;

    protected boolean computedTable = false;

    public LabelProbDP(double knownSum, double [] transitionProbs, double sigmoidPhi){
        this.knownSum = knownSum;
        this.numUnknown = transitionProbs.length;
        range = numUnknown*2 + 1;
        this.transitionProbs = transitionProbs.clone();
        this.phi = sigmoidPhi;
    }

    public double marginal(double l){

        if(numUnknown == 0){
            return this.probL(l , knownSum);
        }

        if(!computedTable){
            this.computeDPTable();
        }

        //now multiply by sigmoid prob and add together
        double marginal = 0.;
        for(int i = 0; i < range; i++){
            double net = net(i);
            double pl = this.probL(l , net);
            double prod = pl * dpTable[i][0];
            marginal += prod;
        }

        return marginal;
    }

    public double logMarginal(double l){

        if(numUnknown == 0){
            return Math.log(this.probL(l , knownSum));
        }

        if(!computedTable){
            this.computeDPTable();
        }

        //now multiply by sigmoid prob and add together

        List<Double> els = new ArrayList<>(range);
        for(int i = 0; i < range; i++){
            double net = net(i);
            double pl = this.probL(l , net);
            double logSum = Math.log(pl) + Math.log(dpTable[i][0]);
            els.add(logSum);
        }

        double logMarginal = logSum(els);

        return logMarginal;
    }

    public void computeDPTable(){

        dpTable =  new double[range][numUnknown]; //all zeros to start

        //first fill in base case last column
        this.fillLastCol();

        //then perform backups
        for(int col = numUnknown-2; col >= 0; col--){

            int maxVal = numUnknown - col;
            int minVal = -maxVal;

            int row0 = row(minVal);
            int rowf = row(maxVal);

            for(int row = row0; row <= rowf; row++){
                dpTable[row][col] = backup(row, col);
            }

        }

        this.computedTable = true;
    }


    public double[][] getDPTable(){
        return dpTable;
    }

    protected void fillLastCol(){

        double plusProb = transitionProbs[numUnknown-1];
        double minusProb = 1. - plusProb;
        dpTable[row(1)][numUnknown-1] = plusProb;
        dpTable[row(-1)][numUnknown-1] = minusProb;

    }


    protected double backup(int row, int col){

        double plusProb = transitionProbs[col];
        double minusProb = 1. - plusProb;

        double plusProd = 0.;
        double minusProd = 0.;

        if(row > 0) {
            plusProd = plusProb * dpTable[row - 1][col + 1];
        }
        if(row < range - 1) {
            minusProd = minusProb * dpTable[row + 1][col + 1];
        }

        double cell = plusProd + minusProd;

        return cell;

    }


    protected int row(double sum){
        int isum = (int)sum;
        int col = isum + numUnknown;
        return col;
    }

    protected int row(int sum){
        int col = sum + numUnknown;
        return col;
    }

    public double net(int row){
        double net = row - numUnknown + knownSum;
        return net;
    }

    protected double probL(double l, double net){
        double sig = sigmoid(net);
        if(l == 1.){
            return sig;
        }
        return 1. - sig;
    }

    protected double sigmoid(double net){
        double denom = 1. + Math.exp(-phi * net);
        double val = 1. / denom;
        return val;
    }

    public static double logSum(List<Double> logEls){
        double max = logEls.get(0);
        for(int i = 1; i < logEls.size(); i++){
            max = Math.max(max, logEls.get(i));
        }

        double expSum = 0.;
        for(double el : logEls){
            double diff = el - max;
            expSum += Math.exp(diff);
        }

        double logSum = Math.log(expSum);
        double res = logSum + max;
        return res;
    }

    public static void main(String[] args) {

        LabelProbDP dp = new LabelProbDP(0, new double[]{0.5, 0.75}, 1.);
        double marginal = dp.marginal(1.);
        double logMarginal = dp.logMarginal(1.);
        System.out.println(marginal);
        System.out.println(Math.exp(logMarginal) + " " + logMarginal);
        System.out.println();
        double [][] table = dp.getDPTable();
        for(int i = 0; i < table.length; i++){
            System.out.println(dp.net(i) + " " + table[i][0]);
        }

    }

}
