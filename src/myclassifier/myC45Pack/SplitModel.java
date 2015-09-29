/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package myclassifier.myC45Pack;

import java.util.Enumeration;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Utils;

/**
 *
 * @author Fahmi
 */
public class SplitModel extends C45ClassifierSplitModel{

    /** Atribut yang akan di-split */
    private int attribIndex;         

    /** Jumlah minimal instance yang akan di-spit   */
    private int minInstances;         

    /** nilai splitPoint yang menentukan split dataSet */
    private double splitPointValue;   

    /** jumlah splitPoints */
    private int numOfSplitPoints;
    
    /** Jumlah cabang yang diinginkan */
    private int numOfBranches;  
  
    /** InfoGain dari node */ 
    private double infoGain; 

    /** GainRatio dari node */
    private double gainRatio;  

    /** total weights dari instances. */
    private double totalWeights;  
    
    public SplitModel(int attribIndex,int minInstances, double totalWeights){
        this.attribIndex = attribIndex;
        this.minInstances = minInstances;
        this.totalWeights = totalWeights;
    }
    public void buildClassifier(Instances dataSet) throws Exception {
        // Initialize the remaining instance variables.
        numSubsets = 0;
        splitPointValue = Double.MAX_VALUE;
        infoGain = 0;
        gainRatio = 0;

        // Different treatment for enumerated and numeric attributes.
        if (dataSet.attribute(attribIndex).isNominal()) {
          numOfBranches = dataSet.attribute(attribIndex).numValues();
          numOfSplitPoints = dataSet.attribute(attribIndex).numValues();
          handleNominalAttribute(dataSet);
        }else{ //attribute numeric
          numOfBranches = 2;
          numOfSplitPoints = 0;
          dataSet.sort(dataSet.attribute(attribIndex));
          handleNumericAttribute(dataSet);
        }
    }
    
    private void handleNominalAttribute(Instances dataSet)
       throws Exception {
    
        Instance instance;
        classDist = new ClassDistribution(numOfBranches,dataSet.numClasses());
        Enumeration instanceEnum = dataSet.enumerateInstances();
        while (instanceEnum.hasMoreElements()){
            instance = (Instance) instanceEnum.nextElement();
            if (!instance.isMissing(attribIndex)){
                classDist.addInstance((int)instance.value(attribIndex),instance);
            }
        }

        // Check if minimum number of Instances in at least two
        // subsets.
        if (classDist.isSplitable(minInstances)) {
          numSubsets = numOfBranches;
          infoGain = classDist.calculateInfoGain(totalWeights);
          gainRatio = classDist.calculateGainRatio(infoGain);
        }
    }
    private void handleNumericAttribute(Instances dataSet)
       throws Exception {
  
        int firstMiss;
        int next = 1;
        int last = 0;
        int splitIndex = -1;
        double currentInfoGain;
        double currentGainRatio;
        double minSplit;
        Instance instance;
        int i;
        boolean instanceMissing = false;

        // Current attribute is a numeric attribute.
        classDist = new ClassDistribution( 2, dataSet.numClasses());

        // Only Instances with known values are relevant.
        Enumeration instanceEnum = dataSet.enumerateInstances();
        i = 0;
        while ((instanceEnum.hasMoreElements() && (!instanceMissing))) {
            instance = (Instance) instanceEnum.nextElement();
            if (instance.isMissing(attribIndex)){
                instanceMissing = true;
            }
            else{
                classDist.addInstance(1,instance);
                i++;
            }
        }
        firstMiss = i;

        // Compute minimum number of Instances required in each
        // subset.
        minSplit =  0.1*(classDist.getTotalWeight())/((double)dataSet.numClasses());
        if (minSplit <= minInstances){ 
            minSplit = minInstances;
        }else if (minSplit > 25){ 
            minSplit = 25;
        }
        // Enough Instances with known values?
        if ((double)firstMiss < 2*minSplit){
            return;
        }
        // Compute values of criteria for all possible split
        // indices.
        //defaultEnt = infoGainCrit.oldEnt(m_distribution);
        while (next < firstMiss) {
          if (dataSet.instance(next-1).value(attribIndex)+1e-5 < 
              dataSet.instance(next).value(attribIndex)){

            // Move class values for all Instances up to next 
            // possible split point.
            classDist.moveInstancesWithRange(1,0,dataSet,last,next);

            // Check if enough Instances in each subset and compute
            // values for criteria.
            if ((classDist.w_perSubdataset[0] >= minSplit) &&
                (classDist.w_perSubdataset[1] >= minSplit)) {
              currentInfoGain = classDist.calculateInfoGain(totalWeights);
              currentGainRatio = classDist.calculateGainRatio(totalWeights);
              if (currentGainRatio >= gainRatio) {
                infoGain = currentInfoGain;
                gainRatio = currentGainRatio;
                splitIndex = next-1;
              }
              numOfSplitPoints++;
            }
            last = next;
          }
          next++;
        }

        // Was there any useful split?
        if (numOfSplitPoints == 0){
            return;
        }
        // Compute modified information gain for best split.
        infoGain = infoGain-(classDist.log2(numOfSplitPoints)/totalWeights);
        if (infoGain > 0){
            // Set instance variables' values to values for
            // best split.
            numSubsets = 2;
            splitPointValue = (dataSet.instance(splitIndex+1).value(attribIndex)+
                dataSet.instance(splitIndex).value(attribIndex))/2;

            // In case we have a numerical precision problem we need to choose the
            // smaller value
            if (splitPointValue == dataSet.instance(splitIndex + 1).value(attribIndex)) {
                splitPointValue = dataSet.instance(splitIndex).value(attribIndex);
            }
            // Restore distributioN for best split.
            classDist = new ClassDistribution(2,dataSet.numClasses());
            classDist.addRange(0,dataSet,0,splitIndex+1);
            classDist.addRange(1,dataSet,splitIndex+1,firstMiss);
            // Compute modified gain ratio for best split.
            gainRatio = classDist.calculateGainRatio(infoGain);
        }
    }
    
    public final void setSplitPoint(Instances allInstances) {
    
        double newSplitPoint = -Double.MAX_VALUE;
        double temp;
        Instance instance;

        if ((allInstances.attribute(attribIndex).isNumeric()) &&
            (numSubsets > 1)) {
          Enumeration instancesEnum = allInstances.enumerateInstances();
          while (instancesEnum.hasMoreElements()) {
            instance = (Instance) instancesEnum.nextElement();
            if (!instance.isMissing(attribIndex)) {
              temp = instance.value(attribIndex);
              if ((temp > newSplitPoint) && (temp <= splitPointValue)){
                newSplitPoint = temp;
              }
            }
          }
          splitPointValue = newSplitPoint;
        }
    }
    @Override
    public double[] getWeights(Instance instance) {
        double [] weights;
        
        if(instance.isMissing(attribIndex)) {
          weights = new double [numSubsets];
          for (int i=0;i<numSubsets;i++){
            weights [i] = classDist.w_perSubdataset[i]/classDist.getTotalWeight();
          }
          return weights;
        }else{
          return null;
        }
    }

    @Override
    public int getSubsetIndex(Instance instance) throws Exception {
        if (instance.isMissing(attribIndex)){
            return -1;
        }else{
            if (instance.attribute(attribIndex).isNominal())
              return (int)instance.value(attribIndex);
            else
              if(instance.value(attribIndex) <= splitPointValue){
                return 0;
              }else{
                return 1;
              }
        }
    }

    @Override
    public String leftSide(Instances data) {
        return data.attribute(attribIndex).name();
    }

    @Override
    public String rightSide(int index, Instances data) {
        StringBuffer text = new StringBuffer();
        if (data.attribute(attribIndex).isNominal()){
          text.append(" = ").append(data.attribute(attribIndex).value(index));
        }else{
          if (index == 0){
            text.append(" <= ").append(Utils.doubleToString(splitPointValue,6));
          }else{
            text.append(" > ").append(Utils.doubleToString(splitPointValue,6));
          }
        }
        return text.toString();
    }
}
