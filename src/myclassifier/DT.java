/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package myclassifier;

import java.util.Random;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.trees.J48;
import weka.core.Instances;
import weka.core.SerializationHelper;
import weka.core.Utils;
import weka.core.converters.ConverterUtils;

/**
 *
 * @author Fahmi
 */
public class DT {
    Classifier DTClassifier;
    Instances data;
    public DT(){
        data = null;
        DTClassifier = new J48();
    }
    public DT(Instances data){
        this.data = data;
        DTClassifier = new J48();
    }
    public void setData(Instances data){
        this.data = data;
    }
    public void CrossValidation() throws Exception{
        if(data!=null){
            Instances train = data;
            // train classifier
            DTClassifier.buildClassifier(train);
            // evaluate classifier and print some statistics
            Evaluation eval = new Evaluation(train);
            eval.crossValidateModel(DTClassifier, train, 10, new Random(1));
            System.out.println(eval.toSummaryString("\nResults\n======\n", false));
            System.out.println(eval.toClassDetailsString("\n=== Detailed Accuracy By Class ===\n"));
            System.out.println(eval.toMatrixString()); 
        }else{
            System.out.println("Data is null");
        }
    }
    public void PercentageSplit(double percent) throws Exception{
        // Percent split
        int trainSize = (int) Math.round(data.numInstances() * percent / 100);
        int testSize = data.numInstances() - trainSize;
        Instances train = new Instances(data, 0, trainSize);
        Instances test = new Instances(data, trainSize, testSize);
        // train classifier
        DTClassifier.buildClassifier(train);
        // evaluate classifier and print some statistics
        Evaluation eval = new Evaluation(train);
        eval.evaluateModel(DTClassifier, test);
        System.out.println(eval.toSummaryString("\nResults\n======\n", false));
        System.out.println(eval.toClassDetailsString("\n=== Detailed Accuracy By Class ===\n"));
        System.out.println(eval.toMatrixString());
    }
    public void SaveModel() throws Exception{
        SerializationHelper.write("DT.model", DTClassifier);
        System.out.println("Model has been saved");
    }
    public void LoadModel() throws Exception{
        DTClassifier = (Classifier) SerializationHelper.read("DT.model");
        System.out.println("Model has been loaded");
    }
    public void Klasifikasi(String filename) throws Exception{
        // load unlabeled data and set class attribute
        Instances unlabeled = ConverterUtils.DataSource.read("unlabeled_"+filename);
        unlabeled.setClassIndex(unlabeled.numAttributes() - 1);
        // create copy
        Instances labeled = new Instances(unlabeled);
        // label instances
        for (int i = 0; i < unlabeled.numInstances(); i++) {
            double clsLabel = DTClassifier.classifyInstance(labeled.instance(i));
            labeled.instance(i).setClassValue(clsLabel);
        }
        // save newly labeled data
        ConverterUtils.DataSink.write("labeled_"+filename, labeled);

        //print hasil
        System.out.println("Classification Result");
        System.out.println("# - actual - predicted - distribution");
        for (int i = 0; i < labeled.numInstances(); i++) {

        double pred = DTClassifier.classifyInstance(labeled.instance(i));
            double[] dist = DTClassifier.distributionForInstance(labeled.instance(i));
            System.out.print((i+1) + " - ");
            System.out.print(labeled.instance(i).toString(labeled.classIndex()) + " - ");
            System.out.print(labeled.classAttribute().value((int) pred) + " - ");
            System.out.println(Utils.arrayToString(dist));
        }
    }
}
