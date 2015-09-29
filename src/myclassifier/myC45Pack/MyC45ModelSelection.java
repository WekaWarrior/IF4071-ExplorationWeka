/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package myclassifier.myC45Pack;

import java.util.Enumeration;
import weka.classifiers.trees.j48.ClassifierSplitModel;
import weka.classifiers.trees.j48.Distribution;
import weka.classifiers.trees.j48.ModelSelection;
import weka.core.Attribute;
import weka.core.Instances;
import weka.core.Utils;

/**
 *
 * @author Fahmi
 */
public abstract class MyC45ModelSelection extends ModelSelection{
    
    /** for serialization */
    private static final long serialVersionUID = 3372204862440821989L;

    /** Minimum number of objects in interval. */
    private int minNumInstance;               

    /** All the training data */
    private Instances allData; // 

    
    public MyC45ModelSelection(int minNumInstance, Instances instances) {
        this.minNumInstance = minNumInstance;
        allData = instances;
    }
    public void cleanup() {
        allData = null;
    }
    
    /*@Override
    public ClassifierSplitModel selectModel(Instances data) throws Exception {
        double minResult;
        double currentResult;
        C45Split [] currentModel;
        C45Split bestModel = null;
        NoSplit noSplitModel = null;
        double averageInfoGain = 0;
        int validModels = 0;
        boolean multiVal = true;
        Distribution checkDistribution;
        Attribute attribute;
        double sumOfWeights;
        int i;

        try{
            // Check if all Instances belong to one class or if not
            // enough Instances to split.
            checkDistribution = new Distribution(data);
            noSplitModel = new NoSplit(checkDistribution);
            if (Utils.sm(checkDistribution.total(), 2*minNumInstance) ||
              Utils.eq(checkDistribution.total(), checkDistribution.perClass(checkDistribution.maxClass()))){
                return noSplitModel;
            }
            
            // Check if all attributes are nominal and have a 
            // lot of values.
            if (allData != null) {
              Enumeration enu = data.enumerateAttributes();
              while (enu.hasMoreElements()) {
                attribute = (Attribute) enu.nextElement();
                if ((attribute.isNumeric()) || (Utils.sm((double)attribute.numValues(),(0.3*(double)allData.numInstances())))){
                    multiVal = false;
                    break;
                }
              }
            }
            
            currentModel = new C45Split[data.numAttributes()];
            sumOfWeights = data.sumOfWeights();

            // For each attribute.
            for (i = 0; i < data.numAttributes(); i++){

              // Apart from class attribute.
              if (i != (data).classIndex()){

                // Get models for current attribute.
                currentModel[i] = new C45Split(i,minNumInstance,sumOfWeights);
                currentModel[i].buildClassifier(data);

                // Check if useful split for current attribute
                // exists and check for enumerated attributes with 
                // a lot of values.
                if (currentModel[i].checkModel())
                  if (allData != null) {
                    if ((data.attribute(i).isNumeric()) ||
                        (multiVal || Utils.sm((double)data.attribute(i).numValues(),
                                              (0.3*(double)allData.numInstances())))){
                      averageInfoGain = averageInfoGain+currentModel[i].infoGain();
                      validModels++;
                    } 
                  } else {
                    averageInfoGain = averageInfoGain+currentModel[i].infoGain();
                    validModels++;
                  }
              }else{
                currentModel[i] = null;
              }
            }

            // Check if any useful split was found.
            if (validModels == 0){
                return noSplitModel;
            }
            averageInfoGain = averageInfoGain/(double)validModels;

            // Find "best" attribute to split on.
            minResult = 0;
            for (i=0;i<data.numAttributes();i++){
                if ((i != (data).classIndex()) && (currentModel[i].checkModel())){
                    // Use 1E-3 here to get a closer approximation to the original implementation.
                    if ((currentModel[i].infoGain() >= (averageInfoGain-1E-3)) && 
                      Utils.gr(currentModel[i].gainRatio(),minResult)){ 
                        bestModel = currentModel[i];
                        minResult = currentModel[i].gainRatio();
                    } 
                }
            }

            // Check if useful split was found.
            if (Utils.eq(minResult,0)){
                return noSplitModel;
            }

            // Add all Instances with unknown values for the corresponding
            // attribute to the distribution for the model, so that
            // the complete distribution is stored with the model. 
            bestModel.distribution().addInstWithUnknown(data,bestModel.attIndex());

            // Set the split point analogue to C45 if attribute numeric.
            if (allData != null){
                bestModel.setSplitPoint(allData);
            }
            
            return bestModel;
        }catch(Exception e){
            e.printStackTrace();
        }
        return null;
        
    }
    */
    @Override
    public String getRevision() {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }
}
