/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package myclassifier;

import java.util.Random;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.core.Instances;
import weka.core.SerializationHelper;
import weka.core.Utils;
import weka.core.converters.ConverterUtils;

/**
 *
 * @author Fahmi
 */
public class naiveBayes {
    Classifier NBClassifier;
    Instances data;
    public naiveBayes(){
        data = null;
        NBClassifier = new NaiveBayes();
    }
    public naiveBayes(Instances data){
        this.data = data;
        NBClassifier = new NaiveBayes();
    }
    public void setData(Instances data){
        this.data = data;
    }
    public void CrossValidation() throws Exception{
        if(data!=null){
            Instances train = data;
            // train classifier
            NBClassifier.buildClassifier(train);
            // evaluate classifier and print some statistics
            Evaluation eval = new Evaluation(train);
            eval.crossValidateModel(NBClassifier, train, 10, new Random(1));
            System.out.println(eval.toSummaryString("\nResults\n======\n", false));
            System.out.println(eval.toClassDetailsString("\n=== Detailed Accuracy By Class ===\n"));
            System.out.println(eval.toMatrixString());
        }else{
            System.out.println("Data is null");
        }
    }
    public void SaveModel() throws Exception{
        SerializationHelper.write("NaiveBayes.model", NBClassifier);
    }
    public void LoadModel() throws Exception{
        NBClassifier = (Classifier) SerializationHelper.read("NaiveBayes.model");
    }
    public void Klasifikasi() throws Exception{
        // load unlabeled data and set class attribute
        Instances unlabeled = ConverterUtils.DataSource.read("unlabeled.arff");
        unlabeled.setClassIndex(unlabeled.numAttributes() - 1);
        // create copy
        Instances labeled = new Instances(unlabeled);
        // label instances
        for (int i = 0; i < unlabeled.numInstances(); i++) {
            double clsLabel = NBClassifier.classifyInstance(labeled.instance(i));
            labeled.instance(i).setClassValue(clsLabel);
        }
        // save newly labeled data
        ConverterUtils.DataSink.write("labeled.arff", labeled);

        //print hasil

        System.out.println("# - actual - predicted - distribution");
        for (int i = 0; i < labeled.numInstances(); i++) {

        double pred = NBClassifier.classifyInstance(labeled.instance(i));
            double[] dist = NBClassifier.distributionForInstance(labeled.instance(i));
            System.out.print((i+1) + " - ");
            System.out.print(labeled.instance(i).toString(labeled.classIndex()) + " - ");
            System.out.print(labeled.classAttribute().value((int) pred) + " - ");
            System.out.println(Utils.arrayToString(dist));
        }
    }
}
