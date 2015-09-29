/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package myclassifier.myC45Pack;

import weka.core.Instance;
import weka.core.Instances;

/**
 *
 * @author Fahmi
 */
public class NoSplitModel extends C45ClassifierSplitModel{
    
    public NoSplitModel(ClassDistribution classDist){
        this.classDist = new ClassDistribution(classDist);
        numSubsets = 1;
    }
    
    @Override
    public double[] getWeights(Instance instance) {
        return null;
    }

    @Override
    public int getSubsetIndex(Instance instance) throws Exception {
        return 0;
    }

    @Override
    public String leftSide(Instances data) {
        return "";
    }

    @Override
    public String rightSide(int index, Instances data) {
        return "";
    }
    
}
