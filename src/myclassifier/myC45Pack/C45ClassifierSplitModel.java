/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package myclassifier.myC45Pack;

import weka.core.Instance;
import weka.core.Instances;
import weka.core.Utils;

/**
 *
 * @author Fahmi
 */
public abstract class C45ClassifierSplitModel {
    /** Distribution of class values. */
    public ClassDistribution classDist;
    /** Number of created subsets. */
    public int numSubsets;
    /* Check model is valid or not*/
    public boolean checkModel(){
        return (numSubsets > 0);
    }
    /**
    * Returns the number of created subsets for the split.
    */
    public int numSubsets(){
      return numSubsets;
    }
    /**
    * Returns weights if instance is assigned to more than one subset.
    * Returns null if instance is only assigned to one subset.
    */
    public abstract double [] getWeights(Instance instance);

    /**
     * Returns index of subset instance is assigned to.
     * Returns -1 if instance is assigned to more than one subset.
     *
     * @exception Exception if something goes wrong
     */
    public abstract int getSubsetIndex(Instance instance) throws Exception;

    public abstract String leftSide(Instances data);
    
    public abstract String rightSide(int index,Instances data);
    
    /**
    * Sets distribution associated with model.
    */
    public void setDistribution(Instances dataSet) throws Exception {
      classDist = new ClassDistribution(dataSet, this);
    }
    /**
    * Gets class probability for instance.
    */
    public double classProb(int classIndex, Instance instance, int subDatasetIndex){
      if (subDatasetIndex > -1) {
        return classDist.prob(classIndex,subDatasetIndex);
      }else {
        double [] weights = getWeights(instance);
        if(weights == null) {
            return classDist.prob(classIndex);
        }else{
            double probability = 0;
            for(int i=0;i<weights.length;i++){
                probability += weights[i] * classDist.prob(classIndex, i);
            }
            return probability;
        }
      }
    }
    /**
    * Prints label for subset index of instances (eg class).
    *
    * @exception Exception if something goes wrong
    */
    public String printLabel(int index,Instances data) throws Exception {

      StringBuffer text = new StringBuffer();
      text.append(((Instances)data).classAttribute().value(classDist.maxClass(index)));
      text.append(" (").append(Utils.roundDouble(classDist.w_perSubdataset[index],2));
      if (Utils.gr(classDist.numIncorrect(index),0)){
        text.append("/").append(Utils.roundDouble(classDist.numIncorrect(index),2));
      }
      text.append(")");

      return text.toString();
    }
    /**
    * Splits the given set of instances into subsets.
    *
    * @exception Exception if something goes wrong
    */
    public Instances [] split(Instances dataSet) throws Exception{ 

      Instances [] newSubsets = new Instances [numSubsets];
      double [] weights;
      double newWeight;
      Instance instance;
      int subset;

      for (int i=0;i<numSubsets;i++){
        newSubsets[i] = new Instances((Instances)dataSet,dataSet.numInstances());
      }
      for (int i = 0; i < dataSet.numInstances(); i++) {
        instance = ((Instances) dataSet).instance(i);
        weights = getWeights(instance);
        subset = getSubsetIndex(instance);
        if (subset > -1){
          newSubsets[subset].add(instance);
        }else{
          for (int j = 0; j < numSubsets; j++){
            if (weights[j] > 0) {
              newWeight = weights[j]*instance.weight();
              newSubsets[j].add(instance);
              newSubsets[j].lastInstance().setWeight(newWeight);
            }
          }
        }
      }
      for (int i = 0; i < numSubsets; i++){
        newSubsets[i].compactify();
      }
      return newSubsets;
    }
  }
