/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package myclassifier;

import weka.classifiers.Classifier;
import weka.core.Instances;

/**
 *
 * @author Fahmi
 */
public class naiveBayes {
    Classifier NBClassifier;
    Instances data;
    public naiveBayes(){
        data = null;
    }
    public naiveBayes(Instances data){
        this.data = data;
    }
}
