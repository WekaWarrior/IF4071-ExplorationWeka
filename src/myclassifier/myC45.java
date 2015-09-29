/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package myclassifier;

import myclassifier.myC45Pack.MyBinC45ModelSelection;
import myclassifier.myC45Pack.MyC45ModelSelection;
import myclassifier.myC45Pack.MyC45PruneableClassifierTree;
import myclassifier.myC45Pack.MyClassifierTree;
import myclassifier.myC45Pack.MyPruneableClassifierTree;
import weka.classifiers.Classifier;
import weka.classifiers.trees.j48.ModelSelection;
import weka.core.Capabilities;
import weka.core.Instance;
import weka.core.Instances;

/**
 *
 * @author Fahmi
 */
public class myC45 extends Classifier {
    
    private MyClassifierTree root;
    private boolean modelUnpruned = false; //unpruned tree or not
    private float CF = 0.25f; //confidence parameter
    private int minNumInstance = 2; //minimum number of instances
    private boolean useLaplace = false; //penggunaan Laplace
    private boolean reducedErrorPruning = false; //penggunaan reduced-error Pruning
    private int numFolds = 3; // number of Folds for reduced-error Pruning
    private int seed = 1; //random number seed untuk reduced-errorPruning
    private boolean binarySplits = false; //binarySplit pada atribut nominal
    private boolean subtreeRaising = true; //penggunaan subtree Raising
    private boolean noCleanup = false; //tidak cleanup setelah tree di-built
    
    
    /*Returns default capabilities of the classifier.*/
    /*public Capabilities getCapabilities() {
      Capabilities      result;

      try {
        if (!reducedErrorPruning)
          result = new MyC45PruneableClassifierTree(null, !modelUnpruned, CF, subtreeRaising, !noCleanup).getCapabilities();
        else
          result = new MyPruneableClassifierTree(null, !modelUnpruned, numFolds, !noCleanup, seed).getCapabilities();
      }
      catch (Exception e) {
        result = new Capabilities(this);
      }

      result.setOwner(this);

      return result;
    }*/
    
//Buat classifier
    @Override
    public void buildClassifier(Instances instances) throws Exception {
        ModelSelection modelSelection;	 
        
        if (binarySplits){
            //modelSelection = new MyBinC45ModelSelection(minNumInstance, instances);
        }else{
            //modelSelection = new MyC45ModelSelection(minNumInstance, instances);
        }
        if (!reducedErrorPruning){
            //root = new MyC45PruneableClassifierTree(modelSelection, !modelUnpruned, CP, subtreeRaising, !noCleanup);
        }else{
            //root = new MyPruneableClassifierTree(modelSelection, !modelUnpruned, numFolds,!noCleanup, seed);
        }
        root.buildClassifier(instances);
        if (binarySplits) {
            //((MyBinC45ModelSelection)modelSelection).cleanup();
        } else {
            //((MyC45ModelSelection)modelSelection).cleanup();
        }
    }
    /**
   * Classifies an instance.
   *
   * @param instance the instance to classify
   * @return the classification for the instance
   * @throws Exception if instance can't be classified successfully
   */
    public double classifyInstance(Instance instance) throws Exception {
      return root.classifyInstance(instance);
    }

    /** 
     * Returns class probabilities for an instance.
     *
     * @param instance the instance to calculate the class probabilities for
     * @return the class probabilities
     * @throws Exception if distribution can't be computed successfully
     */
    public final double [] distributionForInstance(Instance instance) 
         throws Exception {
      return root.distributionForInstance(instance, useLaplace);
    }
   
    public int getSeed() {
        return seed;
    }

    public void setSeed(int newSeed) {
        seed = newSeed;
    }
    public boolean getUseLaplace() {
        return useLaplace;
    }
    public void setUseLaplace(boolean newuseLaplace) {
        useLaplace = newuseLaplace;
    }
    
    public double measureTreeSize() {
        return root.numNodes();
    }

    public double measureNumLeaves() {
        return root.numLeaves();
    }
    
    public boolean getUnpruned() { 
        return modelUnpruned;
    }
    public void setUnpruned(boolean isUnpruned) {
        if (isUnpruned) {
            reducedErrorPruning = false;
        }
        modelUnpruned = isUnpruned;
    }
    
    public float getConfidenceFactor() {
        return CF;
    }
    public void setConfidenceFactor(float newCF) {
        CF = newCF;
    }
    
    public int getMinNumInstance() {
        return minNumInstance;
    }
    public void setMinNumInstance(int newMinNumInst) {
        minNumInstance = newMinNumInst;
    }
    public boolean getReducedErrorPruning() {
        return reducedErrorPruning;
    }
  
    public void setReducedErrorPruning(boolean newREP) {
        if (newREP) {
            modelUnpruned = false;
        }
        reducedErrorPruning = newREP;
    }
    /*Determines the amount of data used for reduced-error pruning. 
     *One fold is used for pruning, the rest for growing the tree.
    */
    public int getNumFolds(){ 
        return numFolds;
    }
    public void setNumFolds(int newNumFolds) {
        numFolds = newNumFolds;
    }
    
    public boolean getBinarySplits() { 
        return binarySplits;
    }
    public void getBinarySplits(boolean isBinSplit) { 
        binarySplits = isBinSplit;
    }
    
    public boolean getSubtreeRaising() {
        return subtreeRaising;
    }
    public void getSubtreeRaising(boolean isSubtreeRaising) {
        subtreeRaising = isSubtreeRaising;
    }
    
    public boolean getSaveInstanceData() { 
        return noCleanup;
    }
    public void setSaveInstanceData(boolean isNoCleanup){
        noCleanup = isNoCleanup;
    }
    
    
}
