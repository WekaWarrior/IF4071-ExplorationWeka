/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package myclassifier.myC45Pack;

import java.util.Enumeration;
import weka.core.Instance;
import weka.core.Instances;

/**
 *
 * @author Fahmi
 */
public class ClassDistribution{
    /** Weight instances setiap class per subdataset. */
    private double[][] w_perClassPerSubdataset;

    /** Weight instances setiap subdataset. */
    double[] w_perSubdataset;

    /** Weight instances setiap class. */
    private double w_perClass[];         

    /** Total weight instances. */
    private double weightTotal;            

    /**
     * Constructor distribution.
     */
    public ClassDistribution(int numSubdataset,int numClasses) {
      w_perSubdataset = new double [numSubdataset];
      w_perClass = new double [numClasses];
      w_perClassPerSubdataset = new double [numSubdataset][numClasses];
      for (int i=0;i<numSubdataset;i++){
        w_perClassPerSubdataset[i] = new double [numClasses];
      }
      weightTotal = 0;
    }


  /**
   * Constructor distribution dengan satu dataset
   * @param dataSet
   * 
   * @exception Exception if something goes wrong
   */
  public ClassDistribution(Instances dataSet) throws Exception {
    w_perClassPerSubdataset = new double [1][dataSet.numClasses()];
    w_perSubdataset = new double [1];
    w_perClass = new double [dataSet.numClasses()];
    weightTotal = 0;
    
    Enumeration E = dataSet.enumerateInstances();
    while (E.hasMoreElements()){
        Instance inst = (Instance) E.nextElement();
        addInstance(0,inst);
    }
  }

  /**
   * Creates a distribution according to given instances and
   * split model.
   *
   * @exception Exception if something goes wrong
   */

  public ClassDistribution(Instances source, C45ClassifierSplitModel modelToUse)
       throws Exception {

    int index;
    Instance instance;
    double[] weights;

    w_perClassPerSubdataset = new double [modelToUse.numSubsets()][0];
    w_perSubdataset = new double [modelToUse.numSubsets()];
    weightTotal = 0;
    w_perClass = new double [source.numClasses()];
    for (int i = 0; i < modelToUse.numSubsets(); i++){
      w_perClassPerSubdataset[i] = new double [source.numClasses()];
    }
    Enumeration E = source.enumerateInstances();
    while (E.hasMoreElements()) {
      instance = (Instance) E.nextElement();
      index = modelToUse.getSubsetIndex(instance);
      if (index != -1){
	addInstance(index, instance);
      }else {
	weights = modelToUse.getWeights(instance);
	addWeights(instance, weights);
      }
    }
  }

  /**
   * Constructor distribution dengan satu dataset
   */
  public ClassDistribution(ClassDistribution sourceDist) {
    w_perClass = new double [sourceDist.getNumClasses()];
    w_perClassPerSubdataset = new double [1][sourceDist.getNumClasses()];
    for(int i=0;i<sourceDist.getNumClasses();i++){
        w_perClassPerSubdataset[0][i] = sourceDist.w_perClass[i];
        w_perClass[i] = sourceDist.w_perClass[i];
    }
    
    weightTotal = sourceDist.weightTotal;
    w_perSubdataset = new double [1];
    w_perSubdataset[0] = weightTotal;
  }

  /**
   * Creates distribution with two bags by merging all bags apart of
   * the indicated one.
   */
  public ClassDistribution(ClassDistribution sourceDist, int index) {
    w_perClass = new double [sourceDist.getNumClasses()];
    //System.arraycopy(sourceDist.w_perClass,0,w_perClass,0,sourceDist.numClasses());
   
    w_perClassPerSubdataset = new double [2][0];
    w_perClassPerSubdataset[0] = new double [sourceDist.getNumClasses()];
    System.arraycopy(sourceDist.w_perClassPerSubdataset[index],0,w_perClassPerSubdataset[0],0,sourceDist.getNumClasses());
    w_perClassPerSubdataset[1] = new double [sourceDist.getNumClasses()];
    for (int i=0;i<sourceDist.getNumClasses();i++){
      w_perClassPerSubdataset[1][i] = sourceDist.w_perClass[i]-w_perClassPerSubdataset[0][i];
      w_perClass[i] = sourceDist.w_perClass[i];
    }
    
    weightTotal = sourceDist.weightTotal;
    w_perSubdataset = new double [2];
    w_perSubdataset[0] = sourceDist.w_perSubdataset[index];
    w_perSubdataset[1] = weightTotal-w_perSubdataset[0];
  }
  
  /**
   * Returns number of non-empty bags of distribution.
   */
  public int getActualNumSubdataset() {
    int returnValue = 0;
    for (int i=0;i<w_perSubdataset.length;i++){
      if (w_perSubdataset[i] > 0){
	returnValue++;
      }
    }
    return returnValue;
  }
  public int getNumSubdataset() {
    return w_perSubdataset.length;
  }
  
  /**
   * Returns number of classes actually occuring in distribution.
   */
  public final int getActualNumClasses() {
    int returnValue = 0;

    for (int i=0;i<w_perClass.length;i++){
      if (w_perClass[i] > 0){
	returnValue++;
      }
    }
    return returnValue;
  }

  /**
   * Returns number of classes actually occuring in given subdataset.
   */
  public final int getActualNumClasses(int subdatasetIndex) {
    int returnValue = 0;
    int i;

    for (i=0;i<w_perClass.length;i++){
      if (w_perClassPerSubdataset[subdatasetIndex][i]>0){
	returnValue++;
      }
    }
    return returnValue;
  }
  
  /**
  * return the number of class in this distribution
  */
  public int getNumClasses(){
    return w_perClass.length;
  }
  
  /**
  * Return the total weight of the class distribution
  */
  public double getTotalWeight(){
    return weightTotal;
  }

  
  /**
   * Adds instance to subDataset.
   *
   * @exception Exception if something goes wrong
   */
  public void addInstance(int subDatasetIndex,Instance instance) 
       throws Exception {
    
    int classIndex = (int)instance.classValue();
    double weight = instance.weight();

    w_perClassPerSubdataset[subDatasetIndex][classIndex] = w_perClassPerSubdataset[subDatasetIndex][classIndex]+weight;
    w_perSubdataset[subDatasetIndex] = w_perSubdataset[subDatasetIndex]+weight;
    w_perClass[classIndex] = w_perClass[classIndex]+weight;
    weightTotal = weightTotal+weight;
  }

  /**
   * Subtracts given instance from given bag.
   *
   * @exception Exception if something goes wrong
   */
  /*public final void sub(int bagIndex,Instance instance) 
       throws Exception {
    
    int classIndex;
    double weight;

    classIndex = (int)instance.classValue();
    weight = instance.weight();
    m_perClassPerBag[bagIndex][classIndex] = 
      m_perClassPerBag[bagIndex][classIndex]-weight;
    m_perBag[bagIndex] = m_perBag[bagIndex]-weight;
    m_perClass[classIndex] = m_perClass[classIndex]-weight;
    totaL = totaL-weight;
  }*/

  /**
   * Adds counts to given bag.
   */
  /*public final void add(int bagIndex, double[] counts) {
    
    double sum = Utils.sum(counts);

    for (int i = 0; i < counts.length; i++)
      m_perClassPerBag[bagIndex][i] += counts[i];
    m_perBag[bagIndex] = m_perBag[bagIndex]+sum;
    for (int i = 0; i < counts.length; i++)
      m_perClass[i] = m_perClass[i]+counts[i];
    totaL = totaL+sum;
  }*/

  /**
   * Adds all instances with unknown values for given attribute, weighted
   * according to frequency of instances in each bag.
   *
   * @exception Exception if something goes wrong
   */
  public void addInstWithMissValue(Instances dataSet,int attIndex)
       throws Exception {

    double [] valueProbs;
    double weight,newWeight;
    int classIndex;
    Instance instance;
    
    valueProbs = new double [w_perSubdataset.length];
    for (int i=0;i<w_perSubdataset.length;i++){
      if (weightTotal == 0){
	valueProbs[i] = 1.0 / valueProbs.length;
      }else{
	valueProbs[i] = w_perSubdataset[i] / weightTotal;
      }
    }
    
    Enumeration E = dataSet.enumerateInstances();
    while (E.hasMoreElements()) {
      instance = (Instance) E.nextElement();
      if (instance.isMissing(attIndex)) {
	classIndex = (int)instance.classValue();
	weight = instance.weight();
	w_perClass[classIndex] = w_perClass[classIndex]+weight;
	weightTotal += weight;
	for (int i = 0; i < w_perSubdataset.length; i++) {
	  newWeight = valueProbs[i] * weight;
	  w_perClassPerSubdataset[i][classIndex] += newWeight;
	  w_perSubdataset[i] += newWeight;
	}
      }
    }
  }
  /**
   * Adds all instances in given range to given bag.
   *
   * @exception Exception if something goes wrong
   */
  /*public final void addRange(int bagIndex,Instances source,
			     int startIndex, int lastPlusOne)
       throws Exception {

    double sumOfWeights = 0;
    int classIndex;
    Instance instance;
    int i;

    for (i = startIndex; i < lastPlusOne; i++) {
      instance = (Instance) source.instance(i);
      classIndex = (int)instance.classValue();
      sumOfWeights = sumOfWeights+instance.weight();
      m_perClassPerBag[bagIndex][classIndex] += instance.weight();
      m_perClass[classIndex] += instance.weight();
    }
    m_perBag[bagIndex] += sumOfWeights;
    totaL += sumOfWeights;
  }
  */
  /**
   * Adds given instance to all bags weighting it according to given weights.
   *
   * @exception Exception if something goes wrong
   */
  public void addWeights(Instance instance, double [] weights)
       throws Exception {

    int classIndex;
    
    classIndex = (int)instance.classValue();
    for (int i=0;i<w_perSubdataset.length;i++) {
      double weight = instance.weight() * weights[i];
      w_perClassPerSubdataset[i][classIndex] += weight;
      w_perSubdataset[i] += weight;
      w_perClass[classIndex] += weight;
      weightTotal += weight;
    }
  }
  
  /**
   * Checks if at least two bags contain a minimum number of instances.
   */
  /*public final boolean check(double minNoObj) {

    int counter = 0;
    int i;

    for (i=0;i<m_perBag.length;i++)
      if (Utils.grOrEq(m_perBag[i],minNoObj))
	counter++;
    if (counter > 1)
      return true;
    else
      return false;
  }
  */
  /**
   * Clones distribution (Deep copy of distribution).
   */
  /*public final Object clone() {

    int i,j;

    Distribution newDistribution = new Distribution (m_perBag.length,
						     m_perClass.length);
    for (i=0;i<m_perBag.length;i++) {
      newDistribution.m_perBag[i] = m_perBag[i];
      for (j=0;j<m_perClass.length;j++)
	newDistribution.m_perClassPerBag[i][j] = m_perClassPerBag[i][j];
    }
    for (j=0;j<m_perClass.length;j++)
      newDistribution.m_perClass[j] = m_perClass[j];
    newDistribution.totaL = totaL;
  
    return newDistribution;
  }
  */
  /**
   * Deletes given instance from given bag.
   *
   * @exception Exception if something goes wrong
   */
  /*public final void del(int bagIndex,Instance instance) 
       throws Exception {

    int classIndex;
    double weight;

    classIndex = (int)instance.classValue();
    weight = instance.weight();
    m_perClassPerBag[bagIndex][classIndex] = 
      m_perClassPerBag[bagIndex][classIndex]-weight;
    m_perBag[bagIndex] = m_perBag[bagIndex]-weight;
    m_perClass[classIndex] = m_perClass[classIndex]-weight;
    totaL = totaL-weight;
  }
  */
  /**
   * Deletes all instances in given range from given bag.
   *
   * @exception Exception if something goes wrong
   */
  /*public final void delRange(int bagIndex,Instances source,
			     int startIndex, int lastPlusOne)
       throws Exception {

    double sumOfWeights = 0;
    int classIndex;
    Instance instance;
    int i;

    for (i = startIndex; i < lastPlusOne; i++) {
      instance = (Instance) source.instance(i);
      classIndex = (int)instance.classValue();
      sumOfWeights = sumOfWeights+instance.weight();
      m_perClassPerBag[bagIndex][classIndex] -= instance.weight();
      m_perClass[classIndex] -= instance.weight();
    }
    m_perBag[bagIndex] -= sumOfWeights;
    totaL -= sumOfWeights;
  }
  */
  /**
   * Prints distribution.
   */
  /*
  public final String dumpDistribution() {

    StringBuffer text;
    int i,j;

    text = new StringBuffer();
    for (i=0;i<m_perBag.length;i++) {
      text.append("Bag num "+i+"\n");
      for (j=0;j<m_perClass.length;j++)
	text.append("Class num "+j+" "+m_perClassPerBag[i][j]+"\n");
    }
    return text.toString();
  }
  */
  /**
   * Sets all counts to zero.
   */
  /*
  public final void initialize() {

    for (int i = 0; i < m_perClass.length; i++) 
      m_perClass[i] = 0;
    for (int i = 0; i < m_perBag.length; i++)
      m_perBag[i] = 0;
    for (int i = 0; i < m_perBag.length; i++)
      for (int j = 0; j < m_perClass.length; j++)
	m_perClassPerBag[i][j] = 0;
    totaL = 0;
  }
  */
  /**
   * Returns matrix with distribution of class values.
   */
  
  /*public final double[][] matrix() {

    return m_perClassPerBag;
  }*/
  
  /**
   * Returns index of bag containing maximum number of instances.
   */
  /*public final int maxBag() {

    double max;
    int maxIndex;
    int i;
    
    max = 0;
    maxIndex = -1;
    for (i=0;i<m_perBag.length;i++)
      if (Utils.grOrEq(m_perBag[i],max)) {
	max = m_perBag[i];
	maxIndex = i;
      }
    return maxIndex;
  }*/

  /**
   * Returns class with highest frequency over all bags.
   */
  public final int maxClass() {

    double max = 0;
    int maxIndex = 0;

    for (int i=0;i<w_perClass.length;i++){
      if (w_perClass[i] > max) {
	max = w_perClass[i];
	maxIndex = i;
      }
    }
    return maxIndex;
  }

  /**
   * Returns class with highest frequency for given subdatas.
   */
  public final int maxClass(int subDatasetIndex) {

    double max = 0;
    int maxIndex = 0;
    int i;

    if (w_perSubdataset[subDatasetIndex] > 0){
      for (i=0;i<w_perClass.length;i++)
	if (w_perClassPerSubdataset[subDatasetIndex][i] > max){
	  max = w_perClassPerSubdataset[subDatasetIndex][i];
	  maxIndex = i;
	}
      return maxIndex;
    }else
      return maxClass();
  }


  /**
   * Returns perClass(maxClass()).
   */
  public final double numCorrect() {

    return w_perClass[maxClass()];
  }

  /**
   * Returns perClassPerSubdataset(index,maxClass(index)).
   */
  public final double numCorrect(int index) {

    return w_perClassPerSubdataset[index][maxClass(index)];
  }

  /**
   * Returns total-numCorrect().
   */
  public final double numIncorrect() {

    return weightTotal-numCorrect();
  }

  /**
   * Returns perBag(index)-numCorrect(index).
   */
  public final double numIncorrect(int index) {

    return w_perSubdataset[index]-numCorrect(index);
  }

  /**
   * Returns number of (possibly fractional) instances of given class in 
   * given bag.
   */
  /*public final double perClassPerBag(int bagIndex, int classIndex) {

    return m_perClassPerBag[bagIndex][classIndex];
  }*/

  /**
   * Returns number of (possibly fractional) instances in given bag.
   */
  /*public final double perBag(int bagIndex) {

    return m_perBag[bagIndex];
  }*/

  /**
   * Returns number of (possibly fractional) instances of given class.
   */
  /*public final double perClass(int classIndex) {

    return m_perClass[classIndex];
  }*/

  /**
   * Returns relative frequency of class over all bags with
   * Laplace correction.
   */
  public double laplaceProb(int classIndex) {

    return (w_perClass[classIndex] + 1) / 
      (weightTotal + (double) w_perClass.length);
  }

  /**
   * Returns relative frequency of class for given bag.
   */
  public double laplaceProb(int classIndex, int intIndex) {

    if (w_perSubdataset[intIndex] > 0){
        return (w_perClassPerSubdataset[intIndex][classIndex] + 1.0) /
             (w_perSubdataset[intIndex] + (double) w_perClass.length);
    }else{
      return laplaceProb(classIndex);
    }
  }

  /**
   * Returns probability of a class
   */
  public double prob(int classIndex) {

    if (weightTotal == 0){
      return w_perClass[classIndex]/weightTotal;
    } else {
      return 0;
    }
  }

  /**
   * Returns relative frequency of class for given subDataset.
   */
  public double prob(int classIndex,int subDatasetIndex) {

    if (w_perSubdataset[subDatasetIndex] > 0){
      return w_perClassPerSubdataset[subDatasetIndex][classIndex]/w_perSubdataset[subDatasetIndex];
    }else{
      return prob(classIndex);
    }
  }

  /** 
   * Subtracts the given distribution from this one. The results
   * has only one bag.
   */
  /*public final Distribution subtract(Distribution toSubstract) {

    Distribution newDist = new Distribution(1,m_perClass.length);

    newDist.m_perBag[0] = totaL-toSubstract.totaL;
    newDist.totaL = newDist.m_perBag[0];
    for (int i = 0; i < m_perClass.length; i++) {
      newDist.m_perClassPerBag[0][i] = m_perClass[i] - toSubstract.m_perClass[i];
      newDist.m_perClass[i] = newDist.m_perClassPerBag[0][i];
    }
    return newDist;
  }*/

  /**
   * Shifts given instance from one bag to another one.
   *
   * @exception Exception if something goes wrong
   */
  /*public final void shift(int from,int to,Instance instance) 
       throws Exception {
    
    int classIndex;
    double weight;

    classIndex = (int)instance.classValue();
    weight = instance.weight();
    m_perClassPerBag[from][classIndex] -= weight;
    m_perClassPerBag[to][classIndex] += weight;
    m_perBag[from] -= weight;
    m_perBag[to] += weight;
  }
  */
  /**
   * Shifts all instances in given range from one bag to another one.
   *
   * @exception Exception if something goes wrong
   */
  /*public final void shiftRange(int from,int to,Instances source,
			       int startIndex,int lastPlusOne) 
       throws Exception {
    
    int classIndex;
    double weight;
    Instance instance;
    int i;

    for (i = startIndex; i < lastPlusOne; i++) {
      instance = (Instance) source.instance(i);
      classIndex = (int)instance.classValue();
      weight = instance.weight();
      m_perClassPerBag[from][classIndex] -= weight;
      m_perClassPerBag[to][classIndex] += weight;
      m_perBag[from] -= weight;
      m_perBag[to] += weight;
    }
  }*/
  
    /**
     * Mengembalikan hasil dari log2
     */
    public double log2(double num) {
        // Constant hard coded for efficiency reasons
        if (num < 1e-6)
          return 0;
        else
          return num*Math.log(num)/Math.log(2);
    }
    
    /**
     * Menghitung entropi
     */
    private double calcInitialEntropy(){
        double initEntropy = 0;
        for(int i=0; i<getNumClasses(); i++){
            double p = w_perClass[i]/weightTotal;
            initEntropy = initEntropy + (p * log2(p));
        }
        return -initEntropy;
    }
    public double calculateInfoGain(double instancesTotalWeight)
    {
        /* initial entropy */
        /* entropy = -(p1 * log2 p1 + p2 * log2 p2 + ...) */
        double initialEntropy = 0;
        double unknownValues = 0;
        double unknownRate = 0;

        initialEntropy = calcInitialEntropy();
        
        for (int i=0; i<getNumSubdataset(); i++){
            double finalEntropy = 0;
            for(int j=0; j<getNumClasses(); j++){
                double p = 0;
                if(w_perSubdataset[i] > 0){
                    p = w_perClassPerSubdataset[i][j] / w_perSubdataset[i];
                }
                finalEntropy = finalEntropy + (p * log2(p));
            }
            finalEntropy = -1*finalEntropy;
            initialEntropy = initialEntropy - (w_perSubdataset[i]/weightTotal*finalEntropy);
        }

        unknownValues = instancesTotalWeight-weightTotal;
        unknownRate = unknownValues/instancesTotalWeight;
        return ((1-unknownRate)*initialEntropy);
    }
    /**
     * Menghitung gain ratio
     * @param infoGain
     * @return
     */
    public double calculateGainRatio(double infoGain) {
        double splitInformation = 0;
        double gainRatio;
        for(int i=0; i<getNumSubdataset(); i++)
        {
            double p = w_perSubdataset[i]/weightTotal;
            splitInformation = splitInformation - (p * log2(p));
        }
        gainRatio = infoGain / splitInformation;
        return gainRatio;
    }
}
