/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package myclassifier;

import myclassifier.myC45Pack.MyBinC45ModelSelection;
import myclassifier.myC45Pack.MyC45ModelSelection;
import myclassifier.myC45Pack.MyC45PruneableClassifierTree;
import myclassifier.myC45Pack.MyPruneableClassifierTree;
import weka.classifiers.Classifier;
import weka.classifiers.trees.j48.ClassifierTree;
import weka.classifiers.trees.j48.ModelSelection;
import weka.core.Instances;

/**
 *
 * @author Fahmi
 */
public class myC45 extends Classifier {
    
    private ClassifierTree root;
    private boolean modelUnpruned = false; //unpruned tree or not
    private float CP = 0.25f; //confidence parameter
    private int minNumInstance = 2; //minimum number of instances
    private boolean useLaplace = false; //penggunaan Laplace
    private boolean reducedErrorPruning = false; //penggunaan reduced-error Pruning
    private int numFolds = 3; // number of Folds for reduced-error Pruning
    private int seed = 1; //random number seed untuk reduced-errorPruning
    private boolean binarySplits = false; //binarySplit pada atribut nominal
    private boolean subtreeRaising = true; //penggunaan subtree Raising
    private boolean noCleanup = false; //tidak cleanup setelah tree di-built
    
    
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
    
}
