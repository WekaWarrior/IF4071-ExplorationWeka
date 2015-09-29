/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package myclassifier;

import java.util.Scanner;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.supervised.instance.Resample;
import weka.filters.unsupervised.attribute.Remove;
/**
 *
 * @author Fahmi
 */
public class MyClassifier {

    /**
     * @param args the command line arguments
     */
    public static void main(String[] args) throws Exception{
        // TODO code application logic here
        double percent;
        
        System.out.println("Silahkan masukkan nama dataset");
        Scanner Read = new Scanner(System.in);
        //String path = Read.next();
        String path = "weather.nominal.arff";
        System.out.println("Nama yg dimasukkan: "+path);
        Instances data = loadData(path);
        
        System.out.println("Choose preprocessing");
        System.out.println("1 = Remove Attrubutes");
        System.out.println("2 = Filter Resample");
        System.out.println("else = Not use it");
        int pilih = Read.nextInt();
        if(pilih==1){
            //remove attributes
            System.out.println("Use Remove Attributes");
            data = removeAttributes(data);
        }else if(pilih==2){
            //filter Resample
            System.out.println("Use Filter Resample");
            System.out.print("Sample Size Percent: ");
            percent = Read.nextDouble();
            data = resampleInstances(data,percent);
        }else{
            System.out.println("Not Use Preprocessing");
        }
        System.out.println("\nMETHOD");
        System.out.println("1. ID3");
        System.out.println("2. Naive Bayes");
        System.out.println("Silahkan pilih method");
        
        pilih = Read.nextInt();
        if(pilih==1){
            DT a = new DT(data);
            //System.out.print("\n*****10 Cross Validation*****");
            //a.CrossValidation();
            System.out.println("Silahkan masukkan persentase untuk percentage split");
            percent = Read.nextDouble();
            a.PercentageSplit(percent);
            System.out.println("\n*****Percentage Split*****");
            a.SaveModel();
            a.LoadModel();
            a.Klasifikasi(path);
        }
        else if(pilih==2){
            naiveBayes b = new naiveBayes(data);
            System.out.print("\n*****10 Cross Validation*****");
            b.CrossValidation();
            /*System.out.println("Silahkan masukkan persentase untuk percentage split");
            percent = Read.nextDouble();
            b.PercentageSplit(percent);
            System.out.print("\n*****Percentage Split*****");*/
            b.SaveModel();
            b.LoadModel();
            b.Klasifikasi(path);
        }
        else{
            System.out.println("Angka yg anda masukkan salah");
        }
    }
    public static Instances loadData(String path) throws Exception{
        DataSource source = new DataSource(path);
        Instances data;
        data = source.getDataSet();
        // setting class attribute if the data format does not provide this information
        // For example, the XRFF format saves the class attribute information as well
        if (data.classIndex() == -1){
            data.setClassIndex(data.numAttributes() - 1);
        }
        return data;
    }
    public static Instances removeAttributes(Instances oldData) throws Exception{
        String[] options = new String[2];
        options[0] = "-R";                                    // "range"
        options[1] = "1";                                     // first attribute
        
        Remove remove = new Remove();                         // new instance of filter
        remove.setOptions(options);                           // set options
        remove.setInputFormat(oldData);                          // inform filter about dataset **AFTER** setting options
        Instances newData = Filter.useFilter(oldData, remove);
        return newData;
    }
    public static Instances resampleInstances(Instances oldData, double sampleSizePercent) throws Exception {
        String Filteroptions="-B 1.0";
        Resample sampler = new Resample();
        sampler.setOptions(weka.core.Utils.splitOptions(Filteroptions));
        sampler.setRandomSeed((int)System.currentTimeMillis());
        sampler.setSampleSizePercent(sampleSizePercent);
        sampler.setInputFormat(oldData);
        Instances newData = Resample.useFilter(oldData,sampler);
        return newData;
    }
}

